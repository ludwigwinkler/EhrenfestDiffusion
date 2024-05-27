# Python
import copy
import dataclasses
import functools as ft
import pyrootutils
import hashlib, time, random

# Bookkeeping
from typing import List, Union
import sys
import numpy as np

import lightning
import matplotlib.pyplot as plt
import timm.scheduler as timm_schedulers

# Machine Learning
import torch, einops
import torch._dynamo
import torchmetrics
import torchvision.utils
from einops import rearrange
from lightning.fabric.utilities.cloud_io import _load as pl_load
from omegaconf import OmegaConf
from pytorch_gan_metrics import get_fid, get_inception_score
from torch import Tensor
from torch.nn.functional import one_hot
from tqdm import tqdm
from .Utils import read_samples_from_directory, check_ckpt
import inspect

torch._dynamo.config.verbose = True

import wandb

torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    print("GPU: ", torch.cuda.get_device_name(0))


class DiscreteDiffusionLightingModule(lightning.LightningModule):
    """data is x(0), noise is x(T)"""

    def __init__(self, cfg, model, criterion, noise_schedule, ctmc):
        super().__init__()
        self.save_hyperparameters(OmegaConf.create(
            dataclasses.asdict(cfg)))  # double trick to get dot access a la hparams.optimization.lr ...

        # compile_fn = ft.partial(torch.compile, backend="aot_eager") if not torch.cuda.is_available() else torch.compile
        self.model = model
        self.ema_model = copy.deepcopy(self.model).requires_grad_(False)
        self.criterion = criterion
        self.noise_schedule = noise_schedule
        self.ctmc = ctmc

        '''Creates a list of global steps when to sample and evaluate over training.'''
        total_sampling_steps = 10
        self.sampling_steps = [(k + 1) * (self.hparams.trainer.max_steps // total_sampling_steps) for k in
                               range(total_sampling_steps)]  # over training, do 20 samplings
        self.sampling_steps = sorted([1_000, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000, self.hparams.trainer.max_steps] + self.sampling_steps)

        self.eval_steps = [(k + 1) * (self.hparams.trainer.max_steps // 5) for k in
                           range(5)]

    def ema_update(self):
        momentum = self.hparams.optimization.ema_momentum
        update = 1 - self.hparams.optimization.ema_momentum
        for param, (ema_param_name, ema_param) in zip(self.model.parameters(), self.ema_model.named_parameters()):
            ema_param.data = momentum * ema_param.data + update * param.data

    def load_checkpoint(self):
        if self.hparams.load_checkpoint:
            try:
                path = pyrootutils.find_root(search_from=__file__, indicator=["config", "src"])
                # run_id_path = f"{self.hparams.checkpoint}" if '/' in self.hparams.checkpoint else f"{self.hparams.checkpoint}/last.ckpt"
                run_id_path = f"{self.hparams.checkpoint}/last.ckpt"
                ckpt_path = str(
                    path) + "/checkpoints/" + run_id_path
                ckpt = check_ckpt(self.hparams, ckpt_path)
                self.load_state_dict(ckpt["state_dict"], strict=True)
                print(f"Loaded checkpoint successfully from {ckpt_path}")
            except Exception as e:
                print(e)
                exit(f"Tried loading checkpoint {ckpt_path} but failed. :(")

    def on_fit_start(self) -> None:

        self.load_checkpoint()

        # self.eval_samples(num_samples=500, datamodule=self.trainer.datamodule)
        # self.trainer.datamodule.eval(num_samples=1_000)
        # exit()

        if self.global_rank == 0:
            batch = next(iter(self.trainer.datamodule.train_dataloader()))
            data = batch.to(self.device)
            print(f"{data.unique().numel()=}")

            # print(f"{data.mean()=} {data.std()=}")
            # sys.exit('on_fit_start')
            t = self.sample_t(data)
            self.noise_schedule.viz_noise_schedule(t, data, show=self.hparams.show, title="Forward")
            # sys.exit()
            self.sample_trajectories(data.to(self.device))
            if hasattr(self.ctmc, "forward_process"):
                gif_path = self.ctmc.forward_process(data=data)
                wandb.log({f"NoiseSchedule/Forward Process": wandb.Video(gif_path)})
            # criterion: dict = self.criterion(model=self.model.eval(), x0=data, t=t, noise_schedule=self.noise_schedule)
            # self.viz_prediction(criterion, title="Training Model")

        # sys.exit()

    @torch.enable_grad()
    def sample_t(self, data: Tensor):
        """Splits [0,T] up to into #batch_size chunks.

        [U(0,T/BS), T/BS * U(0, T/BS), 2*T/BS*U(0, T/BS), ...]
        Achieves consistent and even sampling over [0,T] range
        :param data:
        :return:
        """
        if self.hparams.optimization.weighted_t_sampling and hasattr(self.noise_schedule, "snr"):
            batch_size = data.shape[0]
            bins = 500
            bin_t = torch.linspace(start=0, end=self.hparams.diffusion.T, steps=bins)
            weighted_t = self.noise_schedule.snr(t=bin_t).squeeze()
            plt.plot(bin_t, weighted_t)
            sampled_t = torch.distributions.Categorical(probs=weighted_t).sample((batch_size,)).to(data.device).float()
            sampled_t = sampled_t / bins * self.hparams.diffusion.T
            t = sampled_t + torch.rand((batch_size,), dtype=data.dtype,
                                       device=data.device).float() * self.hparams.diffusion.T / bins
            t = t.clip(0, self.hparams.diffusion.T)
            plt.hist(t)
            plt.show()
            exit()
        else:
            batch_size = data.shape[0]
            t = torch.rand((batch_size,), dtype=data.dtype,
                           device=data.device) * self.hparams.diffusion.T / batch_size
            t = t + (self.hparams.diffusion.T / batch_size) * torch.arange(batch_size, device=data.device)
            # t = t.clamp(min=self.hparams.diffusion.t_min, max=1.)

        t = t.requires_grad_().clamp(0.01, 0.99)
        assert t.numel() == data.shape[0]
        return t.requires_grad_()

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.hparams.optimization.ema_momentum > 0:
            self.ema_update()

    def training_step(self, batch: Union[Tensor, List], batch_idx):
        if len(batch) == 2 and type(batch) == list:
            data = batch[0]
        else:
            data = batch

        assert data.ndim == 4, f"{data.ndim=}"

        t = self.sample_t(data)
        criterion: dict = self.criterion(model=self.model, x0=data, t=t, noise_schedule=self.noise_schedule)

        # if 'pred_dist' in criterion:
        #     plt.plot(criterion['pred_dist'].scale.mean(dim=[1, 2, 3]).detach().cpu().numpy())
        #     plt.show()
        #     exit()
        # print(f"{list(criterion.keys())=}")

        self.log_dict(
            {"loss": criterion["loss"], f"E[{self.model.output_type}]": criterion["output"].mean(),
             "E[x_0]": criterion["x_0"].mean(), "E[target]": criterion['target'].detach().mean(),
             "Std[target]": criterion['target'].detach().std(),
             # 'Target': wandb.Histogram(criterion['target'].detach().cpu().flatten().numpy()),
             # 'Pred': wandb.Histogram(criterion['pred'].detach().cpu().flatten().numpy())},
             },
            prog_bar=True, logger=True)
        # if self.global_step % 1 == 0:  # gotta log it ourselves but not on every step
        # wandb.log({'Target': wandb.Histogram(criterion['target'].detach().cpu().flatten().numpy()),
        #            'Pred': wandb.Histogram(criterion['output'].detach().cpu().flatten().numpy())})
        self.log_dict(
            {"Next Sampling": self.sampling_steps[0] - self.global_step,
             "Next Evaluation": self.eval_steps[0] - self.global_step},
            prog_bar=False,
            logger=True, )
        return criterion["loss"].mean()

    def validation_step(self, batch, batch_idx):
        """Sampling and GIF generation takes some time, so Analytical sampling for epoch=0 NN
        sampling if epoch =1 and then every 10th training epoch."""

        if len(batch) == 2 and type(batch) == list:
            batch = batch[0]
        data = batch
        if batch_idx == 0:
            self.sampling_data = data
        assert data.ndim == 4

        if self.hparams.optimization.ema_momentum > 0:

            model = self.ema_model.to(self.device)
        else:
            model = self.model

        model = model.eval()

        t = self.sample_t(data)
        criterion: dict = self.criterion(model=model, x0=data, t=t, noise_schedule=self.noise_schedule)
        self.log_dict({"Validation/Loss": criterion["loss"].mean()}, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self) -> None:
        if self.global_rank != 0:
            return

        print(f"Global step: {self.global_step=} and remaining sampling steps {self.sampling_steps}")
        if self.global_step >= self.sampling_steps[0]:

            print(f"Sampling at step {self.sampling_steps[0]}")

            data = self.sampling_data
            t = self.sample_t(data)

            criterion: dict = self.criterion(model=self.model.eval(), x0=data, t=t, noise_schedule=self.noise_schedule)
            self.viz_prediction(criterion, title="Training Model")
            self.sample_trajectories(data)
            if self.hparams.optimization.ema_momentum > 0:
                ema_model = self.ema_model.to(self.device).eval()
                ema_criterion: dict = self.criterion(model=ema_model, x0=data, t=t, noise_schedule=self.noise_schedule)
                self.viz_prediction(ema_criterion, title="EMA Model")

            '''
            self.sampling_steps are sorted global training steps when to sample, i.e. [1000, 5000, 10000, 20000, 30000, 40000, 50000]
            If the first element is reached, pop it and sample at the next step. If the list is empty, we're done sampling.
            while loop is necessary to enforce that all elements in the sampling_list are popped ... just in case
            '''
            while self.global_step >= self.sampling_steps[0]:
                self.sampling_steps.pop(0)

        if self.global_step >= self.eval_steps[0]:
            self.eval_samples(num_samples=2_000, datamodule=self.trainer.datamodule)
            while self.global_step >= self.eval_steps[0]:
                self.eval_steps.pop(0)

    def on_test_start(self) -> None:

        print("Testing ...")
        self.sufficient_evaluation_samples = False  # custom early exit for test loop
        self.intermediate_num_test_samples = [1_000, 10_000, 20_000, 30_000, 40_000, 50_000]

        '''Create directory to store samples in'''
        path = pyrootutils.find_root(search_from=__file__, indicator=[".git"])
        path = path / f"experiments/media/ordinal/{self.hparams.checkpoint}/{self.hparams.data.path_to_samples_hash}"
        path.mkdir(parents=True, exist_ok=True)
        self.path_to_sample_directory = path

        if torch.cuda.is_available():
            '''
            Sleeping for a random time to allow staggered testing
            It's a hacky way to get each process to start testing at a different time
            to prevent 10 GPU's sampling an unnecessary final last batch at the same time
            '''
            sleep_time = random.randint(5, 30)
            print(f"Sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)

        if True:
            batch = next(iter(self.trainer.datamodule.train_dataloader()))
            x0 = batch[:36].to(self.device)
            T = self.hparams.diffusion.T
            model = self.ema_model.to(self.device).eval()



            x_t, t, metrics = self.ctmc.sample_trajectories(
                model=model,
                x0=None,
                t=T,
                noise_schedule=self.noise_schedule,
                data_shape=x0.shape,
                T=self.hparams.diffusion.T,
                device=self.device,
            )

            batch_gif_path = self.ctmc.batch_ctmc_gif(
                x_t,
                t,
                title=f"Samples from Model",
                file_str=f"Epoch{self.current_epoch if self.global_step > 0 else -1}",
                # x0_for_statistics=[metrics['x0_t'][-1], x_t[-1]][1]
            )

            wandb.log({f"ReverseProcess/Samples": wandb.Image(plt.gcf())})
            wandb.log({f"NoiseSchedule/Empirical Reverse Moments": wandb.Image(plt.gcf())})
            wandb.log(
                {f"ReverseProcess/{'Analytical' if self.global_step == 0 else ''}SamplingTrajectories": wandb.Video(
                    batch_gif_path)})

            samples_01 = (x_t[-1].float().clamp(-1, 1) + 1) / 2
            fig = plt.figure(figsize=(20, 20))
            samples_viz = torchvision.utils.make_grid(samples_01[:64].cpu(), nrow=6)
            samples_viz = samples_viz.permute(1, 2, 0).numpy()
            plt.imshow(samples_viz)
            plt.title('Samples from Ehrenfest Diffusion Model')
            plt.show()
            exit('on_test_start')


    def check_generated_samples_for_eval(self):
        '''Read all samples that are already stored in directory'''
        test_samples = read_samples_from_directory(directory_path=self.path_to_sample_directory)

        '''Check whether enough samples have already been generated'''
        if test_samples is not None and test_samples.shape[0] > self.hparams.data.num_test_samples:
            test_samples = test_samples.to(self.device)
            eval_dict = self.trainer.datamodule.eval(test_samples[:self.hparams.data.num_test_samples],
                                                     prefix="Test")
            self.log_dict(eval_dict, logger=True)
            self.trainer.should_stop = True
            print(f'Test_samples {test_samples.shape=} are already generated, so stopping')
            print(f"Trainer should stop here ... {self.trainer.should_stop=}")

    def test_step(self, batch, batch_idx):

        if self.trainer.fast_dev_run > 0: sys.exit() # skip laborious testing loop in fast_dev_run

        if len(batch) == 2 and type(batch) == list:
            data = batch[0]
        else:
            data = batch

        if self.hparams.optimization.ema_momentum > 0:
            model = self.ema_model.to(self.device)
        else:
            model = self.model
        model = model.eval()
        S = self.hparams.data.states
        sample_shape = data.shape if torch.cuda.is_available() else (128,) + data.shape[1:]

        '''Lightning can't prematurely exit a test epoch, so check if enough samples have already been generated via a flag'''

        if not self.sufficient_evaluation_samples:
            test_samples = read_samples_from_directory(directory_path=self.path_to_sample_directory)
            # test_samples = torch.randn((3500, 3, 32, 32)).clamp(0, 1).float()
            wandb.log({"Test/NumSamples": test_samples.shape[0] if test_samples is not None else -1})
            while len(self.intermediate_num_test_samples) > 0:
                # iterate through intermediate steps
                print(f"Remaining Intermediate Evaluations: {self.intermediate_num_test_samples}")
                if test_samples is not None and test_samples.shape[0] >= self.intermediate_num_test_samples[0]:
                    # if aren't in general or not enough samples
                    intermediate_num_test_samples = self.intermediate_num_test_samples.pop(0)
                    intermediate_samples = test_samples[:intermediate_num_test_samples]
                    print(
                        f'Evaluating {intermediate_samples.shape[0]} samples (Remaining intermediate steps {self.intermediate_num_test_samples})')
                    eval_dict = self.trainer.datamodule.eval(intermediate_samples.float(), prefix="Test")
                    wandb.log(eval_dict)
                else:
                    # if not test samples or test_samples.shape[0] < self.hparams.data.num_test_samples[0]:
                    print(
                        f'Breaking off intermediate sampling at {test_samples.shape[0] if test_samples is not None else -1} samples and {self.intermediate_num_test_samples} intermediate samples')
                    break

            if test_samples is None or test_samples.shape[0] < self.hparams.data.num_test_samples:
                samples = self.ctmc.sample(T=self.hparams.diffusion.T, device=self.device, data_shape=sample_shape,
                                           model=model)
                samples_01 = (samples.float().clamp(-1, 1) + 1) / 2  # [-S/2, S/2] -> [0, 1]
                samples_hash = hashlib.sha256(samples_01.cpu().numpy().tobytes()).hexdigest()
                samples_path = self.path_to_sample_directory / f"test_samples_{self.hparams.logging.run_id}_{samples_hash[:10]}.pt"
                print(f"Saving samples to {str(samples_path)=}")
                torch.save(samples_01, f=samples_path)

                fig = plt.figure(figsize=(20, 20))
                samples_viz = torchvision.utils.make_grid(samples_01[:64].cpu())
                samples_viz = samples_viz.permute(1, 2, 0).numpy()
                plt.imshow(samples_viz)
                plt.show()
                exit('test_step')
                if test_samples is None or test_samples.shape[0] < 5_000:
                    wandb.log({"Test/Samples": wandb.Image(fig)})
            else:
                self.sufficient_evaluation_samples = True

    def on_test_epoch_end(self) -> None:
        print('On Test Epoch End:')
        samples = read_samples_from_directory(directory_path=self.path_to_sample_directory)
        samples = samples[:self.hparams.data.num_test_samples].to(self.device)
        eval_dict = self.trainer.datamodule.eval(samples, prefix="Test")
        self.log_dict(eval_dict, logger=True)

    @torch.no_grad()
    def eval_samples(self, datamodule=None, num_samples=1_000):
        '''
        Samples from the model and evaluates the samples.
        Intended for quick sampling during training

        '''
        print("Sampling and Evaluating ...")

        if datamodule is None:
            datamodule = self.trainer.datamodule

        data = next(iter(datamodule.train_dataloader()))

        if self.hparams.optimization.ema_momentum > 0:
            model = self.ema_model.to(self.device)
        else:
            model = self.model
        model = model.eval()

        # sample_shape = data.shape if torch.cuda.is_available() else (64,) + data.shape[1:]
        sample_shape = (256,) + data.shape[1:]
        print(f"Eval Sampling Shape: {sample_shape}")
        S = self.ctmc.num_states

        samples = []
        if hasattr(self.ctmc, "sample") and hasattr(datamodule, "eval"):
            for batch_idx in tqdm(range(num_samples // sample_shape[0] + 1), desc="Sampling for Evaluation ..."):
                samples_ = self.ctmc.sample(T=self.hparams.diffusion.T, device=self.device,
                                            data_shape=sample_shape, model=model)
                samples_ = (samples_.cpu().clamp(-1, 1) + 1) / 2  # [-1, 1] -> [0, 1]
                samples += [samples_]

                fig = plt.figure(figsize=(20, 20))
                samples_viz = torchvision.utils.make_grid(samples_[:64].cpu())
                samples_viz = samples_viz.permute(1, 2, 0).numpy()
                plt.imshow(samples_viz)
                plt.title(f'Eval Samples at {self.global_step} Optimization Steps')
                wandb.log(
                    {"Eval/Samples": wandb.Image(fig), "Eval/NumSamples": sum([data.shape[0] for data in samples])})
                plt.show()

            samples = torch.concat(samples, dim=0).to(self.device)[:num_samples]
            print(f"Evaluating with {samples.shape} data")
            eval_dict = datamodule.eval(samples, prefix="Eval")
            wandb.log(eval_dict)
        else:
            print(f"{hasattr(self.ctmc, 'sample')=} and {hasattr(datamodule, 'eval')=}, so no eval ...")

    def viz_prediction(self, data: dict, title=""):
        """Data dict is output from loss function."""

        if "t" in data and "x_t_norm" in data and "x_0" in data:
            idx = slice(0, data["x_0"].shape[0], data["x_0"].shape[0] // 10)
            data_ = {key: data[key].detach().cpu()[idx] for key in ["x_0", "x_t_norm", "x_0_t", "t"]}
            x_0, x_t, pred, t = data_["x_0"], data_["x_t_norm"], data_["x_0_t"], data_["t"]

            fig, axs = plt.subplots(nrows=x_0.shape[0], ncols=3, figsize=(20, 20))

            outlier_percentage = pred[pred.abs() > 1.].shape[0] / pred.numel() * 100

            if outlier_percentage > 5:
                print(f"Outlier Percentage: {pred[pred.abs() > 1].shape[0] / pred.numel() * 100:.2f}%")

            for idx, (x_0_, x_t_, pred_x_0_, t_) in enumerate(zip(x_0, x_t, pred, t)):
                # print(f"{pred_x_0_.min()=} {pred_x_0_.max()=} {x_0.min()=} {x_0.max()=}")
                # print(f"{pred_x_0_.min()=} {pred_x_0_.max()=} {x_t.min()=} {x_t.max()=}")
                if x_0_.shape[0] == 1:
                    axs[idx, 0].matshow(x_0_.permute(1, 2, 0).int().numpy())
                    axs[idx, 1].matshow(x_t_.permute(1, 2, 0).int().numpy())
                    axs[idx, 2].matshow(pred_x_0_.permute(1, 2, 0).int().numpy())
                elif x_0_.shape[0] == 3:
                    # norm_fn = lambda x: torch.clip((x + 1) / 2, min=0., max=1)
                    # norm_fn = lambda x: (x - x.min())/(x.max() - x.min())
                    norm_fn = lambda x: (x.clamp(-1, 1) + 1) / 2
                    axs[idx, 0].imshow(norm_fn(x_0_).permute(1, 2, 0).numpy())
                    axs[idx, 1].imshow(norm_fn(x_t_).permute(1, 2, 0).numpy())
                    axs[idx, 2].imshow(norm_fn(pred_x_0_[:3]).permute(1, 2, 0).numpy())  # [:3] for more than 3 channels
                axs[idx, 0].set_title("x_0")
                axs[idx, 2].set_title("pred_x_0")
                axs[idx, 1].set_title(f"x_t_norm, t={t_.item():.2f}")
            for ax in axs.flatten():
                ax.set_xticks([], [])
                ax.set_yticks([], [])
            plt.tight_layout()
            fig.suptitle(title)
            wandb.log({"Validation/Predictions " + title: wandb.Image(fig)}, commit=False)
            plt.show()
        else:
            for key in ["t", "x_t_norm", "x_0"]:
                print(f"{key} not in dict {list(data.keys())}")

    def sample_trajectories(self, data, num_trajectories=25):
        print(f"Not sampling trajectories in {self.current_epoch+1=}")

        """Use EMA Model"""
        if self.hparams.optimization.ema_momentum > 0:
            model = self.ema_model.to(self.device)
        else:
            model = self.model
        model = model.eval()

        x0 = data[:num_trajectories] if self.global_step == 0 else data[:64]
        T = self.hparams.diffusion.T

        analytical = True
        print(f"{self.global_step=}")
        if self.global_step == 0 and analytical:
            # print('Sampling Analytically ...')
            x_t, t, metrics = self.ctmc.sample_trajectories(
                model=None if self.trainer.fast_dev_run > 0 else model,
                x0=x0 if self.trainer.fast_dev_run > 0 else None,
                t=None,
                noise_schedule=None,
                data_shape=x0.shape,
                T=self.hparams.diffusion.T,
                device=self.device,
            )
        else:
            x_t, t, metrics = self.ctmc.sample_trajectories(
                model=model,
                x0=None,
                t=None,
                noise_schedule=None,
                data_shape=x0.shape,
                T=self.hparams.diffusion.T,
                device=self.device,
            )

        batch_gif_path = self.ctmc.batch_ctmc_gif(
            x_t,
            t,
            title=f"Epoch{self.current_epoch if self.global_step > 0 else -1}",
            file_str=f"Epoch{self.current_epoch if self.global_step > 0 else -1}",
            x0_for_statistics=x0 if self.global_step == 0 else x_t[-1]
        )

        wandb.log({f"ReverseProcess/Samples": wandb.Image(plt.gcf())})
        wandb.log({f"NoiseSchedule/Empirical Reverse Moments": wandb.Image(plt.gcf())})
        # wandb.log({f"ReverseProcess/{'Analytical' if self.global_step == 0 else ''}SamplingTrajectories": wandb.Video(
        #     batch_gif_path)})

        plt.show()

    def configure_optimizers(self):
        if self.hparams.optimization.type == "Adam":
            optim = torch.optim.Adam(self.model.parameters(),
                                     lr=1e-3 if self.hparams.optimization.lr < 0 else self.hparams.optimization.lr)
        if self.hparams.optimization.type == "AdamW":
            optim = torch.optim.AdamW(self.model.parameters(),
                                      lr=1e-3 if self.hparams.optimization.lr < 0 else self.hparams.optimization.lr,
                                      weight_decay=1e-10)

        optimization_dict = {"optimizer": optim}

        if self.hparams.optimization.scheduler:
            if self.hparams.optimization.scheduler_type == "Cosine":
                optimization_dict.update(
                    {
                        "lr_scheduler": {
                            "scheduler": timm_schedulers.CosineLRScheduler(
                                optimizer=optim, t_initial=self.hparams.trainer.max_steps, lr_min=1e-5,
                                warmup_t=self.hparams.optimization.warmup_steps, warmup_lr_init=0.0
                            ),
                            "interval": "step",
                        }
                    }
                )
            if self.hparams.optimization.scheduler_type == "Warmup":
                optimization_dict.update(
                    {
                        "lr_scheduler": {
                            "scheduler": timm_schedulers.StepLRScheduler(optimizer=optim,
                                                                         warmup_t=self.hparams.optimization.warmup_steps,
                                                                         warmup_lr_init=0.0, decay_t=1.0),
                            "interval": "step",
                            "frequency": 1,
                        }
                    }
                )

        return optimization_dict

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.global_step)  # timm's scheduler need the epoch value
