import os
import sys
import traceback

import lightning
import torch
from omegaconf import OmegaConf


def check_ckpt(cfg, ckpt_path):
    """Rudimentary checking that sensible model is loaded."""
    assert os.path.isfile(ckpt_path), f"{ckpt_path} is not a file"
    ckpt = torch.load(ckpt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else cfg.trainer.accelerator))
    '''Comparing configs is kind of difficult due to varying internal data strucutres. :('''
    # if 'hyper_parameters' in ckpt:
        # ckpt_hparams = ckpt["hyper_parameters"]
        # assert cfg.model.type == ckpt_hparams.model.type, f"{cfg.model.type} != {ckpt_hparams.model.type}"
        # if hasattr(ckpt_hparams.model, "residual"):
        #     assert cfg.model.residual == ckpt_hparams.model.residual, f"{cfg.model.residual} != {ckpt_hparams.model.residual}"
        # assert cfg.diffusion.type == ckpt_hparams.diffusion.type, f"{cfg.diffusion.type} != {ckpt_hparams.diffusion.type}"
        # assert cfg.data.type == ckpt_hparams.data.type, f"{cfg.data.type} != {ckpt_hparams.data.type}"
        # assert cfg.data.states == ckpt_hparams.data.states, f"{cfg.data.states} != {ckpt_hparams.data.states}"
    return ckpt


class ExitWithFunctionInfo(Exception):
    def __init__(self, func_name):
        self.func_name = func_name

def read_samples_from_directory(directory_path):
    """Reads all files in a directory and returns a list of them."""
    print(f"Reading files from {directory_path=}")
    file_list = os.listdir(directory_path)
    file_list = [os.path.join(directory_path, file) for file in file_list]
    if len(file_list) == 0:
        print(f'No files found in {directory_path=}')
        return None

    data = []
    for file in file_list:
        if os.path.isfile(file):
            # print(f"File: {file}", sep='')
            data += [torch.load(f=file, map_location='cpu')]
            # print(f"Data: {data[-1].shape}")
    data = torch.cat(data, dim=0).float()
    print(f"{data.min()=} {data.max()=}")
    print(f"Loaded {data.shape=} samples from {len(file_list)} files (Min: {data.min()}, Max: {data.max()})")
    return data