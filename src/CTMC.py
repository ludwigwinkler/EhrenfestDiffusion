# System
import functools
import os
import sys
import warnings
from pathlib import Path

# Bookkeeping
from typing import Union

import einops
import imageio
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.rcParams.update({"axes.grid": True})

# Python
import numpy as np
import pyrootutils
import pytorch_lightning as pl

from typing import Dict, List, Optional, Tuple, Union

# Machine Learning
import torch
import torchvision
import wandb
from einops import rearrange, repeat
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from rich.progress import Progress, track
from torch.distributions import Categorical
from torch.nn.functional import one_hot
from tqdm import tqdm

from . import NoiseSchedules

matplotlib.use("Agg") if torch.cuda.is_available() else None


class EhrenfestCTMC:
    '''
    Tau Leaping:
    Predict a distribution p(x_0 | x_t_norm)
    Compute birth and death rates for each state given every possible state in p(x_0 | x_t_norm) for small \tau
    '''

    def __init__(self, num_states, diffusion, cfg):

        super().__init__()
        self.cfg = cfg
        self.num_states = num_states
        self.hparams_diffusion = diffusion
        self.noise_schedule = getattr(NoiseSchedules, f"{diffusion.type}")(num_states=num_states,
                                                                           cfg=diffusion)

        self.corrector_steps_schedules = {
            'decreasing': lambda t: int(np.round(self.hparams_diffusion.corrector_steps * t)),
            'constant': lambda t: int(self.hparams_diffusion.corrector_steps),
            'increasing': lambda t: int(np.round(
                self.hparams_diffusion.corrector_steps * (1 - t))),
            'nonlin_increasing': lambda t: int(np.round(
                self.hparams_diffusion.corrector_steps * (1 - t ** 0.33)))}

        # self.viz_corrector_schedule()

    def viz_corrector_schedule(self):
        t = np.linspace(0., self.hparams_diffusion.T, 100)

        plt.step(t, [np.clip(self.corrector_steps_schedules[self.hparams_diffusion.corrector_schedule](ti), a_min=0,
                             a_max=self.hparams_diffusion.corrector_steps) for ti in t])
        plt.title(f'Corrector Schedule: {self.hparams_diffusion.corrector_schedule}')
        wandb.log({f"ReverseProcess/CorrectorSchedule": wandb.Image(plt.gcf())})
        plt.show()

    def __repr__(self):
        return (f"Ehrenfest Process with {self.num_states} states")

    @torch.no_grad()
    def sample_trajectories(self, t, T, device, data_shape,
                            x0=None, model=None, noise_schedule=None, sample_aux_data: bool = True):
        # assert (x_0 is not None) or (model is not None), f"x_0 and model is none, currently, x_0: {x_0=} ; model: {model=}"
        assert len(data_shape) == 4
        # assert x0 ^ model, f"{x0=} {model=}"

        if noise_schedule is None:
            noise_schedule = self.noise_schedule

        num_samples, channels, height, width = data_shape
        S_root = 256
        S = S_root ** 2
        # if model: model = model.eval()
        T = 1

        '''Setting up t'''

        t = torch.ones(data_shape, dtype=torch.float32).fill_(T * 0.99).to(device)  # [BS, C, H, W]

        if sample_aux_data:
            tracked_t = [t.clone().detach().cpu()]

        '''Setting up xT'''
        x_T = torch.normal(torch.ones(data_shape, device=device).fill_(0.),
                           torch.ones(data_shape, device=device).fill_(1.)).float()  # .double()
        x_t = x_T
        t = t  # .double()
        if model is not None:
            model = model.eval().float()  # .double()
            device = next(iter(model.parameters())).device
        else:
            device = x0.device

        print(f"{x_t.mean()=} {x_t.std()=}")

        print(f"Sampling {x_T.shape=} {x_T.mean()=} {x_T.std()=}")
        if sample_aux_data:
            tracked_x_t = [x_T.clone().detach().cpu()]
            metrics = {
                "x0_t": [torch.zeros_like(x_t).cpu()],
                "birth_rate": [torch.zeros_like(x_t).cpu()],
                "death_rate": [torch.zeros_like(x_t).cpu()],
            }

        progress_lib = ["tqdm", "rich"][1]
        print()
        if progress_lib == "tqdm":
            pbar = tqdm(total=100, bar_format="{l_bar}{r_bar}")
        else:
            pbar = Progress()
            pbar.start()
            task = pbar.add_task(description="Sampling ", total=None)
        print()

        predictor_steps = 0
        corrector_steps = 0

        print_corrector_start = True

        for step in range(self.hparams_diffusion.max_sampling_steps)[
                    ::-1]:  # counting backwards [999, 998, 997, ... , 0]

            if t.mean() <= 0.01:  # break close to 0
                break

            pred = self.predict(S=S, model=model, t=t, x0=x0, x_t=x_t, noise_schedule=noise_schedule)

            ''' Reverse Ratio Approximation with Score'''
            reverse_death_rate, reverse_birth_rate = noise_schedule.reverse_rates(pred=pred, x_t=x_t, t=t)

            '''Reverse Rates'''
            birth_rate = reverse_birth_rate
            death_rate = reverse_death_rate

            next_state = self.transition(birth_rate=birth_rate, death_rate=death_rate, x_t=x_t,
                                         scaling=torch.scalar_tensor(2 / S_root), device=device)  # .to(device)

            avg_update = (x_t - next_state).abs().mean()
            x_t = next_state
            t = t - 1. / self.hparams_diffusion.max_sampling_steps

            if self.hparams_diffusion.corrector_steps > 0 and self.hparams_diffusion.t_min < t.mean() < self.hparams_diffusion.corrector_schedule_start:
                if print_corrector_start:
                    print(f"Corrector Sampling at {t.mean()=} {self.hparams_diffusion.corrector_schedule=}")
                    print_corrector_start = False
                '''
                t.mean() < 0.99 T: Prevent numerical irregularities close to equilibrium
                '''

                num_corrector_steps = self.corrector_steps_schedules[self.hparams_diffusion.corrector_schedule](
                    t.mean().item())

                if num_corrector_steps > 0:
                    x_t_corr = x_t.clone().detach()
                    # mean_t = t.clone().detach()
                    # t_corr = t.clone().detach()
                    for _ in range(num_corrector_steps):
                        corr_pred = self.predict(S=S, model=model, t=t, x_t=x_t_corr, x0=x0,
                                                 noise_schedule=noise_schedule)

                        forward_birth_rate = noise_schedule.increment_forward_rate(t=t, x_t=x_t_corr)
                        forward_death_rate = noise_schedule.decrement_forward_rate(t=t, x_t=x_t_corr)

                        ''' Reverse Ratio Approximation with Score'''
                        reverse_death_rate, reverse_birth_rate = noise_schedule.reverse_rates(
                            pred=corr_pred,
                            x_t=x_t_corr, t=t,
                        )

                        '''Reverse Corrector Rates'''
                        corr_birth_rate = reverse_birth_rate + forward_birth_rate
                        corr_death_rate = reverse_death_rate + forward_death_rate

                        x_t_corr = self.transition(birth_rate=corr_birth_rate, death_rate=corr_death_rate, x_t=x_t_corr,
                                                   scaling=2 / S_root, device=device)
                        corrector_steps += 1
                    x_t = x_t_corr

            """
            Check next time point and add next state of chain
            BoolTensor.int() -> True becomes 1 and
            """

            ''' Checking Corrector Sampling!'''

            log_freq = 50 if not torch.cuda.is_available() else 50
            if predictor_steps % min(50, max(self.hparams_diffusion.max_sampling_steps // log_freq, 10)) == 0:
                if sample_aux_data:
                    tracked_t += [t.clone().detach().cpu()]
                    tracked_x_t += [x_t.clone().detach().cpu()]

                    metrics["birth_rate"] += [birth_rate.detach().cpu()]
                    metrics["death_rate"] += [death_rate.detach().cpu()]
                wandb.log({'ReverseProcess/ReverseSamplingStep': predictor_steps})

            description = f"{t.mean().item():.3f}/{T}"
            description += f" [{predictor_steps}|{corrector_steps}]"
            description += f"Δx: {avg_update:.2f}"

            description += f" | E[ xt ]: {x_t.mean():.2f} | StD[ xt ]: {x_t.std():.2f}"
            pbar.set_description(description) if progress_lib == "tqdm" else pbar.update(task, refresh=True,
                                                                                         description=description)
            pbar.refresh()

            predictor_steps += 1

        # Final denoising step
        if 'x_0_t' in pred:
            x_t = pred['x_0_t']

        pbar.close() if progress_lib == "tqdm" else pbar.stop()

        if sample_aux_data:
            tracked_x_t += [x_t.clone().detach().cpu()]
            tracked_t += [t.clone().detach().cpu()]
            # metrics["x0_t"] += [x0_t.detach().cpu()]
            metrics["birth_rate"] += [birth_rate.detach().cpu()]
            metrics["death_rate"] += [death_rate.detach().cpu()]

            tracked_x_t = torch.stack(tracked_x_t, axis=0).squeeze(1).cpu()
            tracked_t = torch.stack(tracked_t, axis=0).squeeze(-1).cpu().mean(dim=[2, 3, 4])
            metrics = {key: value for key, value in metrics.items() if
                       len(value) > 0}  # filtering out empty metrics in dictionary
            metrics = {name: torch.stack(val, axis=0).cpu() for name, val in
                       metrics.items()}
            assert tracked_t.dim() == 2 and tracked_t.shape[1] == num_samples, f"{tracked_t.shape=}"

            return tracked_x_t, tracked_t, metrics
        else:
            return x_t.float()

    def sample(self, **kwargs):
        return self.sample_trajectories(**kwargs, t=None, sample_aux_data=False)

    def transition(self, birth_rate, death_rate, x_t, scaling, device):
        if self.hparams_diffusion.sampling_algo == 'bd':
            increment = Categorical(probs=torch.stack([death_rate, birth_rate], dim=-1),
                                    validate_args=False).sample()  # [BS, C, H, W, [death, birth]]
            assert torch.all(increment.unique() == torch.Tensor([0, 1]).to(x_t.device)), f"{increment.unique()=}"
            increment = 2 * increment - 1  # [0, 1] -> [-1, 1]
            next_state = x_t + scaling * increment
            assert birth_rate.shape == death_rate.shape == x_t.shape, f"{birth_rate.shape=} vs {x_t.shape=}"
        elif self.hparams_diffusion.sampling_algo == 'poisson':
            tau = 1 / self.hparams_diffusion.max_sampling_steps
            birth_rate = birth_rate.clamp(min=0.0)
            death_rate = death_rate.clamp(min=0.0)
            assert 0 <= birth_rate.min(), f"{birth_rate.min()=} {birth_rate.max()=}"
            assert 0 <= death_rate.min(), f"{death_rate.min()=} {death_rate.max()=}"
            inc = torch.distributions.Poisson(rate=tau * birth_rate, validate_args=False).sample()
            dec = torch.distributions.Poisson(rate=tau * death_rate, validate_args=False).sample()
            increment = (-dec + inc)  # .clamp(min=-50, max=50)
            next_state = x_t + increment
        elif self.hparams_diffusion.sampling_algo == 'tauleap':
            birth_rate = torch.nan_to_num(birth_rate, nan=0.0, posinf=0.0, neginf=0.0)
            death_rate = torch.nan_to_num(death_rate, nan=0.0, posinf=0.0, neginf=0.0)
            tau = torch.scalar_tensor(1. / self.hparams_diffusion.max_sampling_steps)
            inc = torch.distributions.Poisson(rate=tau * birth_rate.clamp(min=0.001).cpu(),
                                              validate_args=True).sample().to(device)
            dec = torch.distributions.Poisson(rate=tau * death_rate.clamp(min=0.001).cpu(),
                                              validate_args=True).sample().to(device)
            increment = (-dec + inc)  # .clamp(min=-50, max=50)
            next_state = x_t + scaling * increment
            # next_state = next_state.clamp(-S, S)
        return next_state

    def predict(self, S, model, t, x0, x_t, noise_schedule):
        """Predict new x_0."""
        # assert (model is None) or (x_0 is None), "model and x_0 can'tracked_t be both not None, should be XOR"
        if model:
            # x_t_norm = 2.0 * x_t / S  # [-S/2, S/2] -> [-1,1]
            pred = model(x=x_t, t=t)
            pred = model.prob_x0_xt(x=x_t, t=t, S=S,
                                    pred=pred)  # add [BS, C, H, W, S] probabilities to dict if applicable
            if 'epsilon' in pred:
                _, _, std = noise_schedule.compute_moments(t=t, x0=pred['epsilon'])
                w_t = noise_schedule.w_t(t)
                pred['x_0_t'] = (x_t - std * pred['epsilon'].detach()) / w_t
            elif 'score' in pred:
                _, _, std = noise_schedule.compute_moments(t=t, x0=pred['score'])
                w_t = noise_schedule.w_t(t)
                pred['x_0_t'] = (x_t + std.pow(2) * pred['score']).detach() / w_t
            elif 'taylor1' in pred:
                w_t = noise_schedule.w_t(t)
                pred['x_0_t'] = pred['taylor1'].detach() / w_t
        else:  # we're creating artifical predictors with the ground truth
            w_t = noise_schedule.w_t(t)
            _, _, std = noise_schedule.compute_moments(t=t, x0=x0)
            if self.cfg.model.output_type == 'taylor1':
                # x0_norm = 2 * x0 / S  # [-S/2, S/2] -> [-1,1]
                pred = {'taylor1': x0, 'x_0_t': x0}
                x0_t = x0
            elif self.cfg.model.output_type == 'epsilon':
                pred = {'epsilon': (x_t - noise_schedule.w_t(t) * x0) / std, 'x_0_t': x0}
            elif self.cfg.model.output_type == 'taylor2':
                # x0_norm = 2 * x0 / S  # [-S/2, S/2] -> [-1,1]
                pred = {'taylor1': x0, 'taylor2': x0, 'x_0_t': x0, }
                x0_t = x0
            elif self.cfg.model.output_type == 'score_x0':
                _, mu, std = noise_schedule.compute_moments(t=t, x0=x0)
                x0_t = x0
                pred = {'score': - (x_t - mu) / std ** 2, 'x_0_t': x0}
            elif self.cfg.model.output_type in ['gaussian']:

                _, _, std = noise_schedule.compute_moments(t=t, x0=x0)
                gaussian_mu = x0  # ~∈ [-1,1]
                gaussian_scale = (t ** 2).clamp(min=0.01)
                pred = {'gaussian_mu': gaussian_mu, 'gaussian_scale': gaussian_scale}
                pred_mu, pred_scale = pred["gaussian_mu"], pred["gaussian_scale"]
                bin_width = 2. / 256
                bin_centers = torch.linspace(start=-1. + bin_width / 2,
                                             end=1. - bin_width / 2,
                                             steps=int(256),
                                             device=pred_mu.device).view(1, 1, 1, 1, 256)
                bin_centers = torch.ones_like(pred_mu).unsqueeze(-1) * bin_centers
                pred_unsqueezed = torch.distributions.Normal(loc=pred_mu.unsqueeze(-1),
                                                             scale=pred_scale.unsqueeze(-1),
                                                             validate_args=False)
                prob_x0_xt = pred_unsqueezed.log_prob(bin_centers).exp()  # p_θ(x_0 |x_t_norm):[BS, C, H, W, S]
                prob_x0_xt = prob_x0_xt / prob_x0_xt.sum(dim=-1, keepdim=True)
                pred['prob'] = prob_x0_xt
                x0_t = gaussian_mu
            elif self.cfg.model.output_type in ['ratio', 'ratio2']:
                x0_norm = x0  # [-S/2, S/2] -> [-1,1]
                _, _, std_x0 = noise_schedule.compute_moments(t=t, x0=x0)
                var_x0 = std_x0.pow(2)
                birth_rate = torch.exp(-2 * (x_t - noise_schedule.w_t(t) * x0) / (2 * var_x0)).clamp(min=0., max=200)
                death_rate = torch.exp(2 * (x_t - noise_schedule.w_t(t) * x0) / (2 * var_x0)).clamp(min=0., max=200)
                ratio = torch.concat([death_rate, birth_rate], dim=1)
                pred = {'ratio': ratio, 'x_0_t': x0_norm}
                x0_t = x0
            else:
                raise ValueError(f"Unknown type of px0: {self.cfg.model.output_type}")

            pred['output_type'] = self.cfg.model.output_type
            pred['epsilon'] = (x_t - noise_schedule.w_t(t) * x0) / std

        return pred

    def batch_ctmc_gif(self, x_t, t, title="", file_str="", x0_for_statistics=None):
        assert x_t.shape[:2] == t.shape, f"{x_t.shape=} {t.shape=}"
        S = self.num_states
        x_t = x_t
        x_t_01 = (x_t.clamp(-1, 1) + 1) / 2
        x_t_01 = x_t_01.clamp(min=0, max=1)

        plt.hist(x_t[-1].flatten().numpy(), density=True, bins=128)
        plt.title(f'{x_t.shape=} Plotting x_t[-1]')
        plt.show()

        steps, batch, c, h, w = x_t.shape
        if (int(batch ** 0.5) * int(batch ** 0.5)) == batch:
            nrows = int(batch ** 0.5)
            ncols = int(batch ** 0.5)
        else:
            nrows = 5
            ncols = x_t.shape[1] // nrows
            if nrows * ncols < x_t.shape[1]:
                nrows += 1
        figs = []
        if torch.cuda.is_available() and True:
            # on cluster, to not overload logging files
            iterator = range(x_t.shape[0])
            figsize = (10, 10)
        else:
            # on local machine we can have TQDM
            iterator = tqdm(range(x_t.shape[0]), desc="Generating Batch CTMC GIF", bar_format="{l_bar}{r_bar}")

        if x0_for_statistics is not None:
            x0_for_statistics = x0_for_statistics.clone().detach().cpu()
            unique_states = sorted(x0_for_statistics.unique())
            states = unique_states[::(len(unique_states) // 7)]
            colors = matplotlib.colormaps.get_cmap("jet")(torch.linspace(0, 1, len(states)).cpu())
            states_colors = {state: color for state, color in zip(states, colors)}
            mu_t = {state: [] for state in states}
            std_t = {state: [] for state in states}
            t_t = []

        log_freq = 50 if not torch.cuda.is_available() else 50
        for idx in iterator:
            if ((idx % (x_t.shape[0] // np.min([log_freq, x_t.shape[0]])) == 0) or (
                    x_t.shape[0] - idx < 50) or idx < 50) and idx > 0:
                plt.close("all")
                if x0_for_statistics is None:
                    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
                    # axs = np.atleast_1d(axs)
                else:
                    assert x0_for_statistics.shape == x_t[0].shape, f"{x0_for_statistics.shape=} vs {x_t[0].shape=}"
                    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
                if c == 1:
                    grid = einops.rearrange(torchvision.utils.make_grid(x_t_01[idx], nrow=nrows),
                                            "c ... -> ... c").numpy()
                    axs[0].imshow(grid)
                    axs[0].set_title(f"{title}; tracked_t={t[idx].mean():.2f}")
                elif c == 3:
                    assert 0 <= x_t_01[idx].min() and x_t_01[idx].max() <= 1., \
                        f"{x_t_01[idx].min()=} {x_t_01[idx].max()=}"
                    grid = einops.rearrange(torchvision.utils.make_grid(x_t_01[idx], nrow=nrows),
                                            "c ... -> ... c").numpy()
                    if x0_for_statistics is None:
                        plt.imshow(grid)
                        plt.title(f"{title}; tracked_t={t[idx].mean():.2f}")
                    else:
                        axs[0].imshow(grid)
                        axs[0].set_title(f"{title}; tracked_t={t[idx].mean():.2f}")

                if x0_for_statistics is not None:

                    for state in states:
                        filtered_data = x_t[idx][x0_for_statistics == state]
                        mu_t[state] += [filtered_data.mean().item()]
                        std_t[state] += [filtered_data.std().item()]
                    t_t += [t[idx].mean().item()]

                    # print(f"{t_t=} {mu_t[state]=}")

                    fig.suptitle(f"Reverse Process; tracked_t={t[idx].mean():.2f} Std:{x_t[idx].std():.2f}")
                    plt.tight_layout()
                    for state in states:
                        # print(f"{t_t=} {mu_t[state]=}")
                        axs[1].plot(t_t, mu_t[state], color=states_colors[state])
                        min = [mu_t_ - 3 * std_t_ for mu_t_, std_t_ in zip(mu_t[state], std_t[state])]
                        max = [mu_t_ + 3 * std_t_ for mu_t_, std_t_ in zip(mu_t[state], std_t[state])]
                        axs[1].fill_between(x=t_t, y1=min, y2=max, alpha=0.1, color=states_colors[state])
                        axs[1].set_ylim(-3, 3)
                        # axs[1].set_xlim(0, 1.)
                        axs[1].axhline(y=-1, color='black', linestyle="--")
                        axs[1].axhline(y=1, color='black', linestyle="--")
                        # axs[1].grid(True)
                    # axs[1].grid()

                plt.tight_layout()
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                figs += [image]

        path = pyrootutils.find_root(search_from=__file__, indicator=[".git"])
        path = path / f"experiments/media/ordinal/{self.cfg.logging.run_id}"
        path.mkdir(parents=True, exist_ok=True)
        path_gif = path / f"batch_ctmc{file_str}.gif"
        imageio.mimsave(path_gif, figs, "GIF", duration=200)

        if not torch.cuda.is_available() and False:
            def init():
                return []

            def update(frame):
                plt.clf()
                plt.gca().imshow(figs[frame])  # Assuming the figures are stored as images in the list
                # Customize your figure here (labels, titles, etc.)

            import matplotlib.animation as animation

            # fig, ax = plt.subplots()
            ani = animation.FuncAnimation(fig, update, frames=len(figs), init_func=init, repeat=False)
            FFwriter = animation.FFMpegWriter(fps=5)
            path_mp4 = path / f"batch_ctmc{file_str}.mp4"
            ani.save(path_mp4, writer=FFwriter)
            return str(path_mp4)

        return str(path_gif)

    def forward_process(self, data, noise_schedule=None):
        # assert -self.num_states / 2 <= data.min() and data.max() <= self.num_states / 2, f"{data.min()=} {data.max()=}"

        if noise_schedule is None:
            noise_schedule = self.noise_schedule
        # print(f"E[x_T]=0 | Sdt[x_T]={noise_schedule.std_T:.3f}")
        S = self.num_states
        x0 = data
        tracked_x_t = [x0]
        tracked_t = [torch.zeros_like(data)]  # [BS, C, H, W]
        x_t = tracked_x_t[0]
        t = tracked_t[0]

        print('Running Forward Process')

        pbar = Progress()
        pbar.start()
        task = pbar.add_task(description="Sampling ", total=None)

        done = torch.zeros_like(tracked_t[-1]).bool()
        scale = 2 / S
        steps = 0
        T_ = noise_schedule.T
        while not torch.all(done):  # not all(done)
            # x_t_norm[-1] = x_t_norm[-1].clamp(-S/2, S/2)
            # assert -128 <= x_t_norm[-1].min() and x_t_norm[-1].max() <= 128, f"{x_t_norm[-1].min()=} {x_t_norm[-1].max()=}"
            birth_rate = noise_schedule.increment_forward_rate(t=t, x_t=x_t)
            death_rate = noise_schedule.decrement_forward_rate(t=t, x_t=x_t)
            birth_rate_ = birth_rate
            death_rate_ = death_rate

            assert birth_rate.shape == death_rate.shape == x0.shape, f"{birth_rate.shape=} {death_rate.shape=} {x0.shape=}"

            if self.hparams_diffusion.sampling_algo == 'bd':
                increment = Categorical(probs=torch.stack([death_rate, birth_rate], dim=-1)).sample()
                # assert torch.all(increment.unique() == torch.Tensor([0, 1]).to(x_0.device)), f"{increment.unique()=}"
                increment = 2 * increment - 1
                next_state = x_t + scale * increment

                holding_time = self.sample_holding_time(birth_rate=birth_rate, death_rate=death_rate,
                                                        data_shape=birth_rate.shape)
                # if torch.any(t <= (t_close_to_data := 0.05 * torch.ones_like(t))):
                #     holding_time = torch.where(t <= t_close_to_data, holding_time.clip(max=max_holding_time),
                #                                holding_time)
            elif self.hparams_diffusion.sampling_algo in ['tauleap']:
                holding_time = tau = 1 / 1000
                birth_rate = torch.nan_to_num(birth_rate, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=1e-5)
                death_rate = torch.nan_to_num(death_rate, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=1e-5)
                inc = torch.distributions.Poisson(rate=tau * birth_rate.cpu()).sample().to(x0.device)
                dec = torch.distributions.Poisson(rate=tau * death_rate.cpu()).sample().to(x0.device)
                next_state = x_t + scale * (inc - dec)

            done = torch.where(t >= 0.99 * noise_schedule.T, torch.ones_like(done).bool(),
                               torch.zeros_like(done).bool()).bool()
            assert done.shape == data.shape

            '''Update to next state and add holding time'''
            x_t = torch.where(done, x_t, next_state)
            t = t + (~done).float() * holding_time

            '''Save every 50th step for gif to save memory'''
            if steps % 50 == 0 or torch.all(done):
                tracked_t += [t.clone().detach().cpu()]
                tracked_x_t += [x_t.clone().detach().cpu()]
                wandb.log({'ForwardProcess/ForwardSamplingStep': steps})
                wandb.log({'ForwardProcess/ForwardSamplingTime': t.mean().item()})

            description = f"t: {t.mean():.2f} [{steps}] E[x_t_norm]: {x_t.mean():.2f} Std[x_t_norm]: {x_t.std():.2f} "
            # description = f"t: {t.mean():.2f} [{steps}] E[x_t_norm]: {x_t_norm.mean():.2f} Std[x_t_norm]: {x_t_norm.std():.2f} | Min(Rate): {birth_rate_.min(), death_rate_.min()} | Rates<0: {(birth_rate_[birth_rate_ < 0].sum() + birth_rate_[birth_rate_ < 0].sum()) / (2 * birth_rate.numel())}%"
            pbar.update(task, refresh=True, description=description)
            pbar.refresh()
            steps += 1

        pbar.stop()

        tracked_x_t = [x_t_.detach().cpu() for x_t_ in tracked_x_t]
        x0 = x0.detach().cpu()

        diff_x_t = (torch.stack(tracked_x_t) + 1) / 2

        figs = []
        unique_states = sorted(x0.unique())
        states = unique_states[::(len(unique_states) // 7)]
        colors = matplotlib.colormaps.get_cmap("jet")(torch.linspace(0, 1, len(states)).cpu())
        states_colors = {state: color for state, color in zip(states, colors)}
        mu_t = {state: [] for state in states}
        std_t = {state: [] for state in states}
        t_t = []

        pbar = Progress()
        pbar.start()
        task = pbar.add_task(description="Generating ", total=None)
        iterator = range(len(tracked_x_t))
        for idx in iterator:

            if ((idx % (len(tracked_x_t) // np.min([50, len(tracked_x_t)])) == 0) or (
                    len(tracked_x_t) - idx < 20)) or 0 < idx < 10:
                plt.close("all")

                # print(f"{x_t_norm[idx].unique()=}")
                for state in states:
                    filtered_data = tracked_x_t[idx][x0 == state]
                    mu_t[state] += [filtered_data.mean().cpu().detach().numpy()]
                    std_t[state] += [filtered_data.std().cpu().detach().numpy()]
                t_t += [tracked_t[idx].mean().cpu().detach().numpy()]

                # print(f"{x_t_norm[idx].min()=} {x_t_norm[idx].max()=}")

                fig, axs = plt.subplots(1, 3, figsize=(30, 10), gridspec_kw={"width_ratios": [2, 2, 1]})
                norm_fn = lambda x: (x.clamp(-1, 1) + 1) / 2
                grid = torchvision.utils.make_grid(norm_fn(tracked_x_t[idx][:64])).moveaxis(0, -1).clamp(0,
                                                                                                         1.).numpy()
                axs[0].imshow(grid)
                fig.suptitle(
                    f"ForwardDiffusion; tracked_t={tracked_t[idx].mean():.2f} Std:{tracked_x_t[idx].std():.2f}")
                plt.tight_layout()
                for state in states:
                    axs[1].plot(t_t, mu_t[state], color=states_colors[state])
                    min = [mu_t_ - 3 * std_t_ for mu_t_, std_t_ in zip(mu_t[state], std_t[state])]
                    max = [mu_t_ + 3 * std_t_ for mu_t_, std_t_ in zip(mu_t[state], std_t[state])]
                    axs[1].fill_between(x=t_t, y1=min, y2=max, alpha=0.1, color=states_colors[state])
                    axs[1].set_ylim(-3, 3)
                    # axs[1].set_xlim(0, T_)
                    axs[1].axhline(y=-1, color='black', linestyle="--")
                    axs[1].axhline(y=1, color='black', linestyle="--")
                axs[1].grid()
                [ax.grid() for ax in axs.flatten()]
                hist_data = tracked_x_t[idx].detach().float().cpu().flatten().numpy()
                # print(f"{np.unique(hist_data)=}")
                hist, edges = np.histogram(hist_data, bins=100)
                hist = hist.astype(float)
                hist = hist / hist.sum()
                y = (edges[:-1] + edges[1:]) / 2  # y-values represent the bin centers
                # Create a horizontal bar plot
                axs[2].barh(y, hist, height=np.diff(edges), align='center')
                # axs[2].set_ylim(-0.75 * S, 0.75 * S)

                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                figs += [image]

            pbar.update(task, refresh=True, description=f"{idx}/{len(tracked_x_t)}")
            pbar.refresh()

        pbar.stop()

        wandb.log({f"NoiseSchedule/Empirical Forward Moments": wandb.Image(plt.gcf())})
        plt.show()

        path = pyrootutils.find_root(search_from=__file__, indicator=[".git"])
        path = path / f"experiments/media/ordinal/{self.cfg.logging.run_id}"
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"Forward.gif"
        imageio.mimsave(path, figs, "GIF", duration=50)
        return str(path)
