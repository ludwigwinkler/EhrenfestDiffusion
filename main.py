from src.Utils import check_ckpt
from src.Logger import wandb_logger
from src.LightningModule import DiscreteDiffusionLightingModule
from src.Callbacks import get_callbacks
from src import Losses
from src.Losses import EhrenfestVariational, EhrenfestRegression
from config import Config
import src.models
from simple_parsing import ArgumentGenerationMode, ArgumentParser, NestedMode
from src.Logger import flatten_config
from lightning.pytorch.tuner import Tuner
import dataclasses
import functools as ft
import sys
import os.path
import yaml

import lightning
import pyrootutils
import torch

path = pyrootutils.find_root(search_from=__file__, indicator=["config", "src"])
pyrootutils.set_root(
    path=path,  # path to the root directory
    # set the PROJECT_ROOT environment variable to root directory
    project_root_env_var=True,
    dotenv=True,  # load environment variables from .env if exists in root directory
    # add root directory to the PYTHONPATH (helps with imports)
    pythonpath=True,
    # change current working directory to the root directory (helps with filepaths)
    cwd=True,
)
# sys.path.append(str(path))

parser = ArgumentParser(
    argument_generation_mode=ArgumentGenerationMode.BOTH,
    nested_mode=NestedMode.WITHOUT_ROOT,
)

parser.add_arguments(Config, dest="mainconfig")
cfg = parser.parse_args().mainconfig

lightning.seed_everything(cfg.seed)
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('high')

dm = getattr(src.DataModules, f"{cfg.data.type}DataModule")(num_states=cfg.data.states,
                                                            batch_size=cfg.optimization.batch_size,
                                                            resize=cfg.data.resize)

logger = wandb_logger(cfg)
print(cfg)

# with open('config.yaml', 'w') as file:
#     yaml.dump(dataclasses.asdict(cfg), file, default_flow_style=False, sort_keys=True)

# cfg.save("config1.yaml")

# loaded_cfg = Config.load("config2.yaml", drop_extra_fields=False)

# sys.exit()

if "Binary" in cfg.data.type:
    model = ft.partial(getattr(src.models, f"{cfg.model.type}"), in_channels=cfg.data.states,
                       out_channels=cfg.data.states)

    noise_schedule = getattr(
        src.NoiseSchedules, f"Binary{cfg.diffusion.type}")(T=cfg.diffusion.T)
    model = model(output_layer=torch.tanh)
    criterion_str = f"Binary{cfg.criterion.type}"
    ctmc = src.CTMC.BinaryCTMC()
    criterion = getattr(Losses, criterion_str)(classification_weights=dm.classification_weights,
                                               criterionconfig=cfg.criterion)
elif "BirthDeath" in cfg.diffusion.type and "Ordinal" in cfg.data.type:
    model = ft.partial(getattr(src.models, f"{cfg.model.type}"), in_channels=cfg.data.channels,
                       mid_channels=cfg.model.hidden_channels, out_channels=cfg.data.channels,
                       residual=cfg.model.residual)
    model = model()
    noise_schedule_constructor = getattr(
        src.NoiseSchedules, f"{cfg.diffusion.type}")
    forward_noise_schedule = noise_schedule_constructor(
        num_states=cfg.data.states, cfg=cfg.diffusion)
    backward_noise_schedule = noise_schedule_constructor(
        num_states=cfg.data.states, cfg=cfg.diffusion)

    criterion_str = "OrdinalRegression"
    criterion = getattr(Losses, criterion_str)(
        criterion_config=cfg.criterion, model_config=cfg.model)
    ctmc = src.CTMC.BirthDeathCTMC(
        num_states=cfg.data.states, diffusion=cfg.diffusion)
elif "Ehrenfest" in cfg.diffusion.type:
    criterion_str = cfg.criterion.type
    if cfg.model.output_type in ['taylor1', 'taylor2', 'score', 'x0', 'score_x0', 'ratio2', 'epsilon']:
        out_channels = cfg.data.channels
    else:
        out_channels = cfg.data.channels * 2
    model = ft.partial(getattr(src.models, f"{cfg.model.type}"),
                       in_channels=cfg.data.channels,
                       mid_channels=cfg.model.hidden_channels,
                       out_channels=out_channels,
                       residual=cfg.model.residual,
                       dropout=cfg.model.dropout,
                       output_type=cfg.model.output_type)
    model = model()
    noise_schedule_constructor = getattr(
        src.NoiseSchedules, f"{cfg.diffusion.type}")
    forward_noise_schedule = noise_schedule_constructor(
        num_states=cfg.data.states, cfg=cfg.diffusion)
    backward_noise_schedule = noise_schedule_constructor(
        num_states=cfg.data.states, cfg=cfg.diffusion)

    criterion_dict = {'gaussian': EhrenfestVariational, 'logistic': EhrenfestVariational,
                      'ratio': EhrenfestRegression, 'epsilon': EhrenfestRegression, 'taylor1': EhrenfestRegression,
                      'x0': EhrenfestRegression, 'score': EhrenfestRegression, 'ratio2': EhrenfestRegression,
                      'taylor2': EhrenfestRegression}
    criterion = criterion_dict[model.output_type](
        criterion_config=cfg.criterion, model_config=cfg.model)
    ctmc = src.CTMC.EhrenfestCTMC(
        num_states=cfg.data.states, diffusion=cfg.diffusion, cfg=cfg)

elif "VariationalDiscreteCosine" in cfg.diffusion.type and "Ordinal" in cfg.data.type:
    """Variational Classifier on Ordinal Data with Analytical Discrete Diffusion."""
    model = ft.partial(
        getattr(src.models, f"{cfg.model.type}"),
        in_channels=cfg.data.channels,
        mid_channels=cfg.model.hidden_channels,
        out_channels=2 * cfg.data.channels,
        residual=cfg.model.residual,
        output_type="variationaldiscrete",
    )
    forward_noise_schedule = getattr(src.NoiseSchedules, f"{cfg.diffusion.type}")(num_states=cfg.data.states,
                                                                                  cfg=cfg.diffusion)
    backward_noise_schedule = getattr(src.NoiseSchedules, f"{cfg.diffusion.type}")(num_states=cfg.data.states,
                                                                                   cfg=cfg.diffusion)
    model = model()
    criterion_str = "VariationalDiscreteClassification"
    criterion = getattr(Losses, criterion_str)(
        criterion_config=cfg.criterion, model_config=cfg.model)
    ctmc = src.CTMC.VariationalDiscreteCTMC(
        num_states=cfg.data.states, diffusion=cfg.diffusion)
else:
    raise NotImplementedError(f"No valid setup was triggered ...")
    model = ft.partial(
        getattr(src.models, f"{cfg.model.type}"),
        in_channels=cfg.data.states,
        mid_channels=cfg.model.hidden_channels,
        out_channels=cfg.data.states,
        dropout=cfg.model.dropout,
        residual=cfg.model.residual,
        output_type="categorical",
    )
    noise_schedule_constructor = getattr(
        src.NoiseSchedules, f"{cfg.diffusion.type}")
    forward_noise_schedule = noise_schedule_constructor(
        num_states=cfg.data.states, cfg=cfg.diffusion)
    backward_noise_schedule = noise_schedule_constructor(
        num_states=cfg.data.states, cfg=cfg.diffusion)
    model = model()
    criterion_str = f"Discrete{cfg.criterion.type}"
    criterion = getattr(Losses, criterion_str)(
        criterion_config=cfg.criterion, model_config=cfg.model)
    ctmc = src.CTMC.DiscreteCTMC(
        num_states=cfg.data.states, diffusion=cfg.diffusion)

if hasattr(model, "from_pretrained") and cfg.model.load_pretrained:
    model = model.from_pretrained()
else:
    print("Didn't load pretrained")

train_module = DiscreteDiffusionLightingModule(cfg=cfg, model=model, criterion=criterion,
                                               noise_schedule=forward_noise_schedule, ctmc=ctmc)

# train_module = torch.compile(train_module) if cfg.model.compile and torch.cuda.is_available() else train_module
# compile_fn = ft.partial(torch.compile, backend="aot_eager") if not torch.cuda.is_available() else torch.compile
# train_module = compile_fn(train_module) if cfg.model.compile else train_module

# print(dataclasses.asdict(cfg.trainer))
# exit()
trainer = lightning.Trainer(
    **dataclasses.asdict(cfg.trainer), callbacks=get_callbacks(cfg), logger=logger)

if torch.cuda.device_count() == 1 and cfg.optimization.tune_batch_size:
    # if cfg.optimization.tune_batch_size:
    print("Tuning Train Batch Size ...")
    tuner = Tuner(trainer)
    batch_size = tuner.scale_batch_size(model=train_module, datamodule=dm, method="fit", mode="power",
                                        steps_per_trial=3, init_val=64, max_trials=5)
    cfg.optimization.batch_size = batch_size
    train_module.hparams.optimization.batch_size = batch_size
    dm.batch_size = batch_size
    assert next(iter(dm.train_dataloader())).shape[
               0] == batch_size, f"{next(iter(dm.train_dataloader())).shape[0]} vs {batch_size}"
    logger.experiment.config.update({"tuned_batch_size": batch_size})

if cfg.train:
    print("Training ...")
    trainer.fit(model=train_module, datamodule=dm)
else:
    print("Not Training ...")
    if cfg.load_checkpoint:
        try:
            ckpt_path = str(path) + "/checkpoints/" + \
                        f'{cfg.checkpoint}/last.ckpt'
            ckpt = check_ckpt(cfg, ckpt_path)
            train_module.load_state_dict(ckpt["state_dict"], strict=True)
            print(f"Loaded checkpoint successfully from {ckpt_path}")
        except Exception as e:
            print(e)
            exit(f"Tried loading checkpoint {ckpt_path} but failed. :(")
    else:
        print(f"Not loading checkpoint {cfg.checkpoint}...")

if torch.cuda.device_count() == 1 and cfg.optimization.tune_batch_size:
    print("Tuning Test Batch Size ...")
    tuner = Tuner(trainer)
    """Sampling is only performed at end of validation epoch, so first 4 steps per trial do not
    trigger sampling method."""
    batch_size = tuner.scale_batch_size(model=train_module, datamodule=dm, method="validate", mode="power",
                                        steps_per_trial=4, init_val=64, max_trials=4)
    cfg.optimization.batch_size = batch_size
    train_module.hparams.optimization.batch_size = batch_size
    dm.batch_size = batch_size

# train_module.eval_samples(num_samples=10_000, datamodule=dm)
# trainer.validate(model=train_module, datamodule=dm)

trainer.test(model=train_module, datamodule=dm)
# train_module.eval_samples(num_samples=50_000, datamodule=dm)

# trainer.test(model=train_module, dataloaders=dm.val_dataloader()) # using for val_dataloader for more samples
