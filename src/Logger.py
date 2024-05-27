import os, sys
import random

import pyrootutils
import simple_parsing
import wandb
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
import hashlib, json, copy
import torch
import yaml, dataclasses
import itertools

# have to import it like this, cause Omegaconf's strict typechecking will detect "wrong" type with longer path in front
# e.g. DiscreteDiffusion.config.OptimizationConfig != config.OptimizationConfig for OemgaConf
sys.path.append('../config')
from config import ModelConfigs, DataConfigs, CriterionConfig, TrainerConfig, LoggingConfig, \
    OptimizationConfig


# from config.DataClassConfig import ModelConfigs, DataConfigs, CriterionConfig, TrainerConfig, LoggingConfig, OptimizationConfig

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


def config_to_hash(cfg):
    dict_json_str = json.dumps(cfg, sort_keys=True)
    cfg_hash_object = hashlib.sha256(dict_json_str.encode())
    hash_code = str(cfg_hash_object.hexdigest())[:10]
    return hash_code


def wandb_logger(cfg):
    os.environ["WANDB__SERVICE_WAIT"] = "90"
    """https://lightning.ai/docs/pytorch/stable/extensions/generated/pytorch_lightning.loggers.Wand
    bLogger.html#

    -> Log Model checkpoints
    describes how model checkpoints can be saved to wandb cloud
    """
    path = pyrootutils.find_root(search_from=__file__, indicator=["config", "src"])

    if cfg.seed == -1:
        cfg.seed = random.randint(0, 2 ** 32 - 1)

    hash_cfg_dict = flatten_config(copy.deepcopy(cfg))

    hash_cfg_dict.pop('seed')
    cfg.hparams_hash = config_to_hash(hash_cfg_dict)

    hash_cfg_dict.pop('data.num_eval_samples')
    cfg.path_to_samples_hash = config_to_hash(hash_cfg_dict)

    if cfg.model.output_type in ['ratio', 'ratio2']:
        cfg.diffusion.rate == 'ratio'  # ratios require ratios

    '''
    Load *ONLY* Checkpoint Model Config into current config 
    '''
    if cfg.load_checkpoint:
        config_path = path / f"checkpoints/{cfg.checkpoint}/{cfg.checkpoint}.yaml"
        print(f"{cfg=}")
        # print(f"{config_path=}")
        # loaded_config = cfg.load(str(config_path), drop_extra_fields=False)
        with open(str(config_path), 'r') as stream:
            loaded_config = yaml.safe_load(stream)
        print_config(cfg)
        print(loaded_config['model'])
        print(loaded_config['diffusion'])
        model_cfg = ModelConfigs[loaded_config['model']['type']](**loaded_config['model'])
        optimization_cfg = OptimizationConfig(**loaded_config['optimization'])
        cfg.optimization = optimization_cfg
        cfg.model = model_cfg

        cfg.diffusion.schedule_type = loaded_config['diffusion']['schedule_type']

        print(f"Updated Model Config:")
        print_config(cfg)


    logger = WandbLogger(
        mode=cfg.logging.mode,
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        config=flatten_config(cfg),
        # log_model='all',
        tags=[],
    )

    args = logger.experiment.settings._args
    command = " ".join(args)

    """Pull unique hash-type ID from logger to identify checkpoints quickly."""
    logger.experiment.config.update(
        {"logging.run_id": logger.version, "command": command}, allow_val_change=True)
    if torch.cuda.is_available():
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        slurm_array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
        slurm_array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        logger.experiment.config.update({'slurm_job_id': str(slurm_job_id),
                                         'slurm_array_job_id': str(slurm_array_job_id) + "_" + str(
                                             slurm_array_task_id)},
                                        allow_val_change=True)

    cfg.logging.run_id = logger.version if logger.version is not None or logger.version != '' else 'test_debug'

    wandb.run.log_code(root=path, include_fn=lambda path_: path_.endswith(".py"))

    if torch.cuda.is_available():
        print("GPU: ", torch.cuda.get_device_name(0))

    config_path = path / f"checkpoints/{cfg.logging.run_id}"
    config_path.mkdir(parents=True, exist_ok=True)
    cfg.save(config_path / f"{cfg.logging.run_id}.yaml")

    return logger
