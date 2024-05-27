# System
import copy
import dataclasses
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pyrootutils
import simple_parsing
import torch
from simple_parsing import ArgumentGenerationMode, ArgumentParser, NestedMode, Serializable, choice, subgroups
from simple_parsing.helpers import FlattenedAccess, list_field

path = pyrootutils.find_root(search_from=__file__, indicator=[".pre-commit-config.yaml"])
pyrootutils.set_root(
    path=path,  # path to the root directory
    project_root_env_var=True,  # set the PROJECT_ROOT environment variable to root directory
    dotenv=True,  # load environment variables from .env if exists in root directory
    pythonpath=True,  # add root directory to the PYTHONPATH (helps with imports)
    cwd=True,  # change current working directory to the root directory (helps with filepaths)
)

TrainerConfigs = {}
DataConfigs = {}
DiffusionConfigs = {}
ModelConfigs = {}

subconfigs = simple_parsing.subgroups


# def flatten_config(cfg):
#     return pd.json_normalize(dataclasses.asdict(cfg), sep=".").to_dict(orient="records")[0]


def flatten_config(config):
    def abs_path_nested_dict(dictionary, hparams_dict: dict, my_keys=""):
        for key, value in dictionary.items():
            current_key = my_keys + "." + key
            if type(value) is dict:
                abs_path_nested_dict(value, hparams_dict, current_key)
            else:
                hparams_dict.update({current_key[1:]: value})

    nested_config_dict = simple_parsing.helpers.serialization.to_dict(config)
    flattened_config_dict = {}
    abs_path_nested_dict(nested_config_dict, flattened_config_dict)
    return flattened_config_dict


# def flatten_config(cfg):
# 	return flatdict.FlatDict(cfg, delimiter='.')


def compute_equilibrium_std(S):
    """
    At equilibrium:
    forward_mu = S/2
    0 = forward_mu - 3 forward_std = S/2 - 3 forward_std
                                                                    -> forward_std = S/6
    :param S:
    :return: terminal_std
    """
    return S / 6


def compute_lambda_T(S):
    """
    At equilibrium:
    forward_mu = S/2
    0 = forward_mu - 3 forward_std = S/2 - 3 forward_std
                                                                    -> forward_std = S/6
    var_T = = forward_std^2 2 lambda_T -> forward_std^2 = 2 lambda_T
                                                                    -> lambda_T = (S/6)^2 /2
    For 256:
                    mu_T = S/2 = 128
                    0 = S/2 - 3 * std_T -> std_T = S/6 = 42.667
                    std_T^2 = var_T = 2 lambda_T -> lambda_T = std_T^2/2
    :param S:
    :return: terminal_std
    """
    return (S / 6) ** 2 / 2


def compute_lambda_0(std=6):
    """
    At equilibrium:
    var_T = = std_0^2 = 2 lambda_0
    std_0^2 = 2 lambda_0
    lambda_0 = forward_std^2 /2
    :param S:
    :return: terminal_std
    """
    return (std) ** 2 / 2


def print_config(config):
    flattened_config = flatten_config(config)

    iterator = itertools.groupby(flattened_config.items(), lambda keyvalue: keyvalue[0].split(".")[0])

    print()
    print("Config:")
    for i_key, (key, group) in enumerate(copy.deepcopy(iterator)):
        print(key)
        right_tabs = max([len(key__) for key__, _ in copy.deepcopy(group)]) + 1
        for key_, val in group:
            print(f"\t {key_:{right_tabs}} : {type(val).__name__}\t = {val}")
    print()


def register(registry):
    """Registers constructor in dictionary."""

    def store_in_register(config):
        """Automatically uses name of class and wraps it in a dataclass."""
        registry[config.__name__] = dataclass(config)
        return config

    return store_in_register


@dataclass
class NicePrintDataClass:

    def __str__(self):
        flattened_config = flatten_config(self)

        iterator = itertools.groupby(flattened_config.items(), lambda keyvalue: keyvalue[0].split(".")[0])

        ungrouped_length = []
        for i_key, (key, group) in enumerate(copy.deepcopy(iterator)):
            len_group = len(list(copy.deepcopy(group)))
            if not len_group > 1:
                ungrouped_length.append(len(key))
        max_ungrouped_length = max(ungrouped_length)
        str = ""
        str += "\nCONFIG\n"
        for i_key, (key, group) in enumerate(copy.deepcopy(iterator)):
            len_group = len(list(copy.deepcopy(group)))
            if len_group > 1:
                str += f"{key.upper()} \n"
            right_tabs = max([len(key__) for key__, _ in copy.deepcopy(group)]) + 1
            for key_, val in group:
                if len_group > 1:
                    str += f"\t {key_:{right_tabs}} : {type(val).__name__}\t = {val} \n"
                else:
                    str += f"{key_:{max_ungrouped_length}} : {type(val).__name__} = {val} \n"
        return str


@dataclass
class DataConfig:
    data_dir: str = str(path / "data")
    num_eval_samples: int = 3_000
    path_to_samples_hash: str = ""


@register(DataConfigs)
class BinaryMNIST(DataConfig):
    type: str = "BinaryMNIST"
    resize = (32, 32)
    states: int = 256
    channels: int = 3


@register(DataConfigs)
class DiscreteMNIST(DataConfig):
    type: str = "DiscreteMNIST"
    resize: Tuple[int] = (32, 32)
    states: int = 19
    channels: int = 1


@register(DataConfigs)
class Cityscapes(DataConfig):
    type: str = "Cityscapes"
    resize: Tuple[int] = (32, 64)
    states: int = 19
    channels: int = 1


@register(DataConfigs)
class MNIST(DataConfig):
    type: str = "MNIST"
    resize = (32, 32)
    states: int = 256  # subtract one since 0 is modeled as first state
    channels: int = 3


@register(DataConfigs)
class CIFAR10(DataConfig):
    type: str = "CIFAR10"
    resize: Tuple[int] = (32, 32)
    states: int = 256  # subtract one since 0 is modeled as first state
    channels: int = 3
    num_eval_samples: int = 10_000
    num_test_samples: int = 50_000


@dataclass
class TrainerConfig:
    """Arguments for Trainer, passed as kwargs to Trainer.init()"""

    max_steps: int = 500_000
    devices: int = 1
    accelerator: str = ["mps", "gpu"][int(torch.cuda.is_available())]
    devices: int = torch.cuda.device_count() if torch.cuda.is_available() else 1
    limit_train_batches: float = 100 if not torch.cuda.is_available() else 1.0
    limit_val_batches: float = 20 if not torch.cuda.is_available() else 1.0
    limit_test_batches: float = 20 if not torch.cuda.is_available() else 1.0
    num_sanity_val_steps: int = 1 if not torch.cuda.is_available() else 0
    check_val_every_n_epoch: int = 1 if not torch.cuda.is_available() else 1  # we manually run validation based on steps
    log_every_n_steps: int = 50 if not torch.cuda.is_available() else 50
    enable_progress_bar: bool = True if not torch.cuda.is_available() else False
    fast_dev_run: int = 0
    precision: str = ["32", "16-mixed", "bf16-mixed"][0 if not torch.cuda.is_available() else 1]
    gradient_clip_val: float = 1.0  # default algo is norm
    inference_mode: bool = False


@dataclass
class OptimizationConfig:
    lr: float = 1e-3 if not torch.cuda.is_available() else 2e-4
    type: str = "Adam"
    batch_size: int = 36 if not torch.cuda.is_available() else 256
    tune_batch_size: bool = False
    scheduler: bool = True if not torch.cuda.is_available() else True
    scheduler_type: str = ["Cosine", "Warmup"][1]
    warmup_steps: int = 50 if not torch.cuda.is_available() else 5_000
    weighted_t_sampling: bool = False
    ema_momentum: float = -1 if not torch.cuda.is_available() else 0.9999  # momentum * ema_param + (1 - momentum) * param


@dataclass
class LoggingConfig:
    mode: str = ["disabled", "online"][0]
    entity: str = "ludwigwinkler"
    project: str = "ScaledEhrenfest"
    run_id: str = ""


@dataclass
class CriterionConfig:
    type: str = "EhrenfestRegression"
    normalize: bool = [False, True][1]
    elbo_weight: float = 1.0
    denoise_weight: float = 1.0


@dataclass
class ModelConfig:
    compile: bool = False
    residual: bool = False


@register(ModelConfigs)
class DiffuserUNet:
    type: str = "DiffuserUNet"
    load_pretrained: bool = True if not torch.cuda.is_available() else True
    residual: bool = True
    hidden_channels: int = 128 if not torch.cuda.is_available() else 128
    dropout: float = 0.1
    output_type: str = {0: 'gaussian', 2: 'taylor1', 3: 'taylor2',
                        4: 'ratio', 5: 'x0', 6: 'score', 7: 'score_x0', 8: 'epsilon', 9: 'taylor1', 10: 'ratio2'}[4]


@register(ModelConfigs)
class PyTorchDDPM(ModelConfig):
    type: str = "PyTorchDDPM"
    load_pretrained: bool = False if not torch.cuda.is_available() else False
    residual: bool = True
    hidden_channels: int = 128 if not torch.cuda.is_available() else 128
    dropout: float = 0.1
    output_type: str = {0: 'gaussian', 2: 'taylor1', 3: 'taylor2',
                        4: 'ratio', 5: 'x0', 6: 'score', 7: 'score_x0', 8: 'epsilon', 9: 'taylor1', 10: 'ratio2'}[10]


@register(ModelConfigs)
class ImprovedDDPM(ModelConfig):
    type: str = "ImprovedDDPM"
    load_pretrained: bool = True if not torch.cuda.is_available() else True
    residual: bool = True
    hidden_channels: int = 128 if not torch.cuda.is_available() else 128
    dropout: float = 0.0
    output_type: str = {0: 'gaussian', 2: 'taylor1', 3: 'taylor2',
                        4: 'ratio', 5: 'x0', 6: 'score', 7: 'score_x0', 8: 'epsilon', 9: 'taylor1', 10: 'ratio2'}[6]


@register(ModelConfigs)
class SmallUNet:
    type: str = "Small"
    load_pretrained: bool = False
    hidden_channels: int = -1
    residual: bool = True


@register(ModelConfigs)
class DDPM(ModelConfig):
    type: str = "DDPM"


@dataclass
class DiffusionProcess:
    T: float = 1.0
    w_reg: float = 0.0
    w_max: float = 1.0
    softmax_momentum: float = -1.0


@register(DiffusionConfigs)
class DiscreteManfredDiffusion(DiffusionProcess):
    type: str = "Manfred"
    T: float = 1.0
    freeze_target: float = 0.00
    w_reg: float = 0.0


@register(DiffusionConfigs)
class DiscreteCosineDiffusion(DiffusionProcess):
    type: str = "DiscreteCosineWeight"
    T: float = 1.0
    freeze_target: float = 0.00
    w_reg: float = 0.0


@register(DiffusionConfigs)
class VariationalDiscreteCosineDiffusion(DiffusionProcess):
    type: str = "VariationalDiscreteCosineWeight"
    T: float = 1.0
    freeze_target: float = 0.00
    w_reg: float = 0.0


@register(DiffusionConfigs)
class DiscreteCosineRootDiffusion(DiffusionProcess):
    type: str = "CosineRootWeight"
    T: float = 1.0
    freeze_target: float = 0.00
    w_reg: float = 0.0


@register(DiffusionConfigs)
class ForwardBirthDeath(DiffusionProcess):
    type: str = "SeparateBirthDeath"
    T: float = 1.0
    schedule_type: str = ["power", "campbell"][0]
    power: float = 1.0
    lambda_T: float = [compute_lambda_T(S=256), 0.][0]  # Equilibrium Normal distribution
    lambda_0: float = 1.  # Noising Rate at t=0
    freeze_target: float = 0.0
    stationary_dist: str = ["gaussian", "uniform"][0]
    stationary_scale: float = compute_equilibrium_std(256)


@register(DiffusionConfigs)
class BackwardBirthDeath(ForwardBirthDeath):
    # schedule_type = 'power'
    schedule_type: str = ["power", "campbell"][0]
    # power: float = 2
    lambda_0: float = 12
    # lambda_T: float = [compute_lambda_T(S=256), 1000][1]
    ratio_target: str = ["x_0", "mu_t"][0]
    max_sampling_steps: int = 2000
    min_sampling_steps: int = -1
    corrector_steps: int = 0
    corrector_start: float = 0.1
    stationary_scale: float = compute_equilibrium_std(S=256)
    stationary_dist = "gaussian"


@register(DiffusionConfigs)
class TimeGardinerBirthDeath(DiffusionProcess):
    type: str = "TimeGardinerBirthDeath"
    T: float = 1.0
    schedule_type: str = "power"
    power: float = 1.0
    theta_0: float = 0.
    theta_T: float = 10.
    lambda_T: float = 0.  # Equilibrium Normal distribution
    lambda_0: float = 0.  # Noising Rate at t=0
    ratio_target: str = ["x_0", "mu_t"][0]
    max_sampling_steps: int = 2000
    min_sampling_steps: int = 200
    corrector_steps: int = 0
    corrector_start: float = 0.1
    stationary_dist: str = ["gaussian", "uniform"][0]
    stationary_scale: float = compute_equilibrium_std(256)


@register(DiffusionConfigs)
class TimeEhrenfest(DiffusionProcess):
    type: str = "TimeEhrenfest"
    T: float = 1.0
    t_min: float = 0.01
    schedule_type: str = "cosine"
    lambda_T: float = -1.  # Equilibrium Normal distribution
    lambda_0: float = 0.  # Noising Rate at t=0
    epsilon_T: float = 3500
    max_sampling_steps: int = 2000 if not torch.cuda.is_available() else 2000
    min_sampling_steps: int = -1
    corrector_steps: int = 0
    corrector_schedule: str = ['decreasing', 'constant', 'increasing', 'nonlin_increasing'][1]
    corrector_postmax: bool = False
    corrector_schedule_start: float = 0.1
    increment_clamp_t: float = -1
    increment_clamp: int = 100
    final_denoise: bool = [False, True][1]
    sampling_algo: str = ['bd', 'tauleap'][1]


@register(DiffusionConfigs)
class ScaledEhrenfest(DiffusionProcess):
    type: str = "ScaledEhrenfest"
    T: float = 1.0
    t_min: float = 0.01
    schedule_type: str = ["cosine", 'song'][1]
    rate: str = ['taylor1', 'score', 'ratio'][-1]
    song_beta_min: float = 0.1
    song_beta_max: float = 20
    max_sampling_steps: int = 1000 if not torch.cuda.is_available() else 1000
    corrector_steps: int = 0
    corrector_schedule: str = ['decreasing', 'constant', 'increasing', 'nonlin_increasing'][1]
    corrector_schedule_start: float = 0.1
    final_denoise: bool = [False, True][1]
    sampling_algo: str = ['bd', 'tauleap'][1]


@dataclass
class Config(FlattenedAccess, Serializable):
    show: bool = [False, True][1]
    hparams_hash: str = ""
    seed: int = 12345
    train: bool = [False, True][0]
    checkpoint: str = ["r4ftq2q0"][-1]
    load_checkpoint: bool = False
    sample_every_n_optimization_steps: int = 100 if not torch.cuda.is_available() else 10_000
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    criterion: CriterionConfig = field(default_factory=CriterionConfig)
    data: DataConfig = subgroups(DataConfigs, default='CIFAR10')
    model: ModelConfig = subconfigs(ModelConfigs, default='PyTorchDDPM')
    diffusion: DiffusionProcess = subconfigs(DiffusionConfigs, default='ScaledEhrenfest')


if __name__ == "__main__":
    parser = ArgumentParser(
        argument_generation_mode=ArgumentGenerationMode.BOTH,
        nested_mode=NestedMode.WITHOUT_ROOT,
    )

    parser.add_arguments(Config, dest="mainconfig")
    cfg = parser.parse_args().mainconfig

    print(cfg)
    if cfg.logging.mode == "disabled":
        cfg.trainer.show_progress_bar = True

    print(cfg)

    # print(dataclasses.asdict(Config()))
