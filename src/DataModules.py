import os
from pathlib import Path

import imageio
import pyrootutils
import torchmetrics.image

path = pyrootutils.find_root(search_from=__file__, indicator=[".git"])
pyrootutils.set_root(
    path=path,  # path to the root directory
    project_root_env_var=True,  # set the PROJECT_ROOT environment variable to root directory
    dotenv=True,  # load environment variables from .env if exists in root directory
    pythonpath=True,  # add root directory to the PYTHONPATH (helps with imports)
    cwd=True,  # change current working directory to the root directory (helps with filepaths)
)

from typing import Optional, Tuple

import einops
import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from .pytorch_gan_metrics_local import get_fid, get_inception_score, get_inception_score_and_fid
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

# from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10, MNIST, Cityscapes
from tqdm.auto import tqdm


class ToyDataDataModule(pl.LightningDataModule):
    def __init__(self, num_states, batch_size=128, resize=[32, 32]):
        super().__init__()


class MNIST(MNIST):
    def __getitem__(self, item):
        return super().__getitem__(item)[0]


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, num_states, data_dir: str = path / "data", batch_size=128, resize=[32, 32]):
        super().__init__()
        self.num_states = num_states
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.transform = None

        self.prepare_data()

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def label_filter(self, dataset, labels):
        mask = torch.isin(dataset.targets, torch.Tensor(labels))
        dataset.data = dataset.data[mask]
        dataset.targets = dataset.targets[mask]

    def setup(self, stage: str = "fit"):
        # Assign train/val datasets for use in dataloaders

        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            # self.label_filter(mnist_full, labels=[0, 2, 3, 4, 5, 6, 7, 8, 9])
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self, batch_size=None):
        num_workers = 2 if torch.cuda.is_available() else 0
        return DataLoader(self.mnist_train, batch_size=batch_size or self.batch_size, shuffle=True,
                          num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)


class BinaryMNISTDataModule(MNISTDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(self.hparams.resize, antialias=True),
             transforms.Lambda(lambda x: (x > 0.5) * 1.0), transforms.Lambda(lambda x: 2 * x - 1)]
        )

        self.setup()
        self.classification_weights, self.sample_weights = self.compute_classification_rebalancing_weights()

    def compute_classification_rebalancing_weights(self):
        pos, neg = 0, 0
        for i, batch in tqdm(enumerate(self.train_dataloader(batch_size=512))):
            data = batch[0]
            pos += data[data > 0].numel()
            neg += data[data < 0].numel()
        num_class_samples = torch.Tensor([neg, pos])
        sample_weights = num_class_samples / num_class_samples.sum()
        classification_weights = 1 / (2 * sample_weights)
        print(f"{sample_weights} -> weights: {classification_weights}")
        # exit()
        return classification_weights, sample_weights


class DiscreteMNISTDataModule(MNISTDataModule):
    """MNIST is [C=1, H, W] and we want [S, H, W] for S>1."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        assert self.hparams.num_states >= 2, f"{self.hparams.num_states=} is not >=2"
        boundaries = torch.linspace(start=0, end=1, steps=self.hparams.num_states)
        """
		torch.bucketize: [1, H, W] -> [c, H, W]

		"""
        self.transform = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                torchvision.transforms.Resize(self.hparams.resize, antialias=True),
                torchvision.transforms.Lambda(lambda data: torch.bucketize(data, boundaries)),
                torchvision.transforms.Lambda(
                    lambda data: torch.nn.functional.one_hot(data, num_classes=self.hparams.num_states)),
                torchvision.transforms.Lambda(lambda data: einops.rearrange(data, "c h w s -> s h w c").squeeze(-1)),
                torchvision.transforms.Lambda(lambda data: data.float()),
            ]
        )

        self.setup()
        self.prepare_data()

    # self.classification_weights, self.sample_weights = self.compute_classification_rebalancing_weights()

    def compute_classification_rebalancing_weights(self):
        path = pyrootutils.find_root(search_from=__file__, indicator=[".git"])
        path = path / "data/datastatistics/"
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"discrete{self.hparams.num_states}_classificationsampleweights.pt"

        if not os.path.isfile(path):
            num_class_samples = torch.zeros(self.hparams.num_states)
            for i, batch in tqdm(enumerate(self.train_dataloader(batch_size=512))):
                data = batch
                num_class_samples += data.sum(dim=[0, 2, 3])
            sample_weights = num_class_samples / num_class_samples.sum()
            classification_weights = 1 / (2 * sample_weights)
            torch.save({"classification_weights": classification_weights, "sample_weights": sample_weights}, f=path)
        else:
            classification_weights, sample_weights = torch.load(path).values()
        print(f"{sample_weights} -> weights: {classification_weights}")
        return classification_weights, sample_weights

    def eval(self, data):
        assert data is not None

        if not hasattr(self, "data_probability"):
            path = pyrootutils.find_root(search_from=__file__, indicator=[".git"])
            path = path / "data/datastatistics/"
            path.mkdir(parents=True, exist_ok=True)
            path = path / f"discrete{self.hparams.num_states}_mnistpixelprobs.pt"

            if not os.path.isfile(path):  # if not on file, we need to compute it
                data_ = []
                label = []
                for i, (data__, label__) in tqdm(enumerate(self.train_dataloader(batch_size=512))):
                    data_ += [data__]
                    label += [label__]
                data_ = torch.cat(data_, dim=0).double()
                label = torch.cat(label, dim=0).double()
                target_idx = torch.arange(0, label.numel()).reshape_as(label)

                """
				Mean and forward_std in [Subset_Class, C, H, W]
				torch.nan_to_num(tensor, value)
				target_idx[label == subset_target_] filters out all target_idx's where label==subset_target_
				"""
                data_frequency = torch.stack(
                    [data_[target_idx[label == subset_target_]].sum(dim=0) + 1.0 for subset_target_ in
                     label.unique()]).double()
                data_probability = data_frequency / (data_frequency.sum(dim=1, keepdim=True) + 0.0)
                assert torch.isnan(data_probability).sum() == 0
                dist = torch.distributions.Categorical(data_probability)
                torch.save({"data_probability": data_probability}, f=path)
            else:  # file exists and we load it
                data_probability = torch.load(f=path)["data_probability"]
                dist = torch.distributions.Categorical(data_probability)

            self.data_probability = einops.rearrange(data_probability, "l s h w -> l h w s")  # l = label, s = state

            if False:
                for i, label in enumerate(self.data_probability):
                    fig, axs = plt.subplots(1, 3)
                    axs = axs.flatten()
                    label = einops.rearrange(label, "h w c -> c h w")
                    for label_state, ax in zip(label, axs):
                        img = ax.matshow(label_state, vmin=0, vmax=1)
                        ax.set_xticks([], [])
                        ax.set_yticks([], [])
                    fig.suptitle(f"{i}")
                    plt.show()

        # data_probability = einops.repeat(self.data_probability, 'l h w s -> b l h w s',
        # 								 b=data.shape[0])  # add batch dimension
        data_ = einops.rearrange(data, "b s h w -> b h w s")  # move states to last dim
        data_ = einops.repeat(data_, "b h w s -> b l h w s", l=self.data_probability.shape[0])  # add label dimension
        dist = torch.distributions.Categorical(self.data_probability)
        """Data:[BS, Label, H, W, S] -> argmax()->[BS, Label, H, W] log_prob."""
        log_prob = dist.log_prob(data_.argmax(dim=-1)).mean(dim=[-2, -1])

        if 0:
            fig, axs = plt.subplots(4, 4, figsize=(10, 10))
            axs = axs.flatten()
            for data_, log_prob_, ax in zip(data[:16], log_prob[:16], axs):
                # print(f"{data_.shape=}")
                ax.matshow(data_.argmax(dim=0))
                ax.set_title(f"{log_prob_.argmax()}({log_prob_.max().item():.2f})")
            plt.tight_layout()
            plt.show()
            exit()
        return {"Sampling/LogProb": log_prob}


class OrdinalMNISTDataModule(MNISTDataModule):
    """MNIST is [C=1, H, W] and we want [S, H, W] ∈ [0, S]."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.S = 256
        self.transform = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                torchvision.transforms.Resize(self.hparams.resize, antialias=True),
                torchvision.transforms.Lambda(
                    lambda data: 2 / self.S * (
                            torch.round(self.S * einops.repeat(data.float(), '1 ... -> 3 ...')) - self.S // 2)),
                # torchvision.transforms.Lambda(lambda data: data - self.num_states / 2)
            ]
        )

        self.setup()
        self.prepare_data()


class CIFAR10(CIFAR10):
    def __getitem__(self, item):
        return super().__getitem__(item)[0]


class BaseCIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, num_states, data_dir: str = path / "data", batch_size=128, resize=[32, 32]):
        super().__init__()
        self.num_states = num_states
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.transform = None

        self.num_workers = 4 * torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"Using {self.num_workers=} in CIFAR10DataModule")

        self.prepare_data()
        self.setup()

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def label_filter(self, dataset, labels):
        mask = torch.isin(dataset.targets, torch.Tensor(labels))
        dataset.data = dataset.data[mask]
        dataset.targets = dataset.targets[mask]

    def setup(self, stage: str = "fit"):
        # Assign train/val datasets for use in dataloaders

        if stage == "fit":
            self.cifar10_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar10_val = CIFAR10(self.data_dir, train=False, transform=self.transform)
        # self.label_filter(mnist_full, labels=[0, 2, 3, 4, 5, 6, 7, 8, 9])
        # self.cifar10_train, self.cifar10_val = random_split(mnist_full, [0.9, 0.1])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.cifar10_test = CIFAR10(self.data_dir, train=True, transform=self.transform)

        if stage == "predict":
            self.cifar10_predict = CIFAR10(self.data_dir, train=True, transform=self.transform)

    def train_dataloader(self, batch_size=None):
        return DataLoader(self.cifar10_train, batch_size=batch_size or self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=512, num_workers=self.num_workers, drop_last=False)

    def predict_dataloader(self):
        return DataLoader(self.cifar10_predict, batch_size=self.batch_size)


class CIFAR10DataModule(BaseCIFAR10DataModule):
    """CIFAR10 is [0, 1]"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.S = self.hparams.num_states - 1
        self.transform = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                torchvision.transforms.Resize(self.hparams.resize, antialias=True),
                transforms.RandomHorizontalFlip(),
                # torchvision.transforms.Lambda(lambda data: 2 * data - 1),
                torchvision.transforms.Lambda(
                    lambda data: 2 / self.S * (torch.round(self.S * data.float()) - self.S // 2)),
            ]
        )
        self.vanilla_transform = transform_lib.Compose([transform_lib.ToTensor(),
                                                        torchvision.transforms.Resize(self.hparams.resize,
                                                                                      antialias=True)])

        self.setup()
        self.prepare_data()

    def eval(self, data=None, prefix="Eval", num_samples=1000):
        """FID needs to be fed both with real data and.

        :param data:
        :return:
        """

        print('CIFAR10 Eval Inception Score')
        if data is None:
            data = []
            dataloader = DataLoader(CIFAR10(self.data_dir, train=True, transform=self.vanilla_transform),
                                    batch_size=512)
            for batch_data in dataloader:
                data += [batch_data + torch.randn_like(batch_data) * 0.0]
            data = torch.concat(data, dim=0)[:num_samples]
            print(f"{data.min()=} {data.max()=}")
            dataloader = DataLoader(data, batch_size=512, drop_last=False)

        else:
            assert 0. <= data.min() and data.max() <= 1., f"{data.min()=} {data.max()=}"
            data = data.float()
            dataloader = DataLoader(data, batch_size=512, drop_last=False, num_workers=0)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

        # IS, IS_std = get_inception_score(dataloader, device=device, use_torch=True, verbose=True)
        # print(f"{IS=} {IS_std=}")
        (IS, IS_std), FID = get_inception_score_and_fid(
            dataloader, self.data_dir / 'fid_cifar10.test.npz', device=device, use_torch=False, verbose=True)
        print(f"Samples {data.shape[0]}: {IS=} {IS_std=} {FID=}")
        return {f"{prefix}/IS": IS.item(), f"{prefix}/IS_std": IS_std.item(), f"{prefix}/FID": FID.item(),
                f"{prefix}/NumEvalSamples": data.shape[0]}


class Cityscapes(Cityscapes):
    def __getitem__(self, index: int):
        """Only get segmentation mask."""
        return super().__getitem__(index)[1].float()


class CityscapesDataModule(pl.LightningDataModule):
    def __init__(self, num_states, data_dir: str = path / "data/cityscapes", batch_size=128, resize=[32, 32]):
        super().__init__()
        self.save_hyperparameters()
        self.num_states = num_states
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize

        unimportant_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        important_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 255]
        if num_states is not None or num_states > 0:
            important_classes = important_classes[:num_states]
        print(f"Unimportant Classes: {len(unimportant_classes)}")
        print(f"Important Classes: {len(important_classes)}")
        map_class_to_zero_dict = {key: key for key in range(-1, 255)}
        map_class_to_zero_dict.update({key: 0 for key in unimportant_classes})
        map_class_to_zero_dict.update({key: idx for idx, key in enumerate(important_classes)})

        boundaries = torch.linspace(start=0, end=num_states, steps=len(important_classes))

        self.transform = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                torchvision.transforms.Lambda(lambda data: torch.bucketize(data, boundaries)),
                torchvision.transforms.Lambda(
                    lambda data: torch.nn.functional.one_hot(data, num_classes=self.hparams.num_states)),
                torchvision.transforms.Lambda(lambda data: einops.rearrange(data, "c h w s -> s h w c").squeeze(-1)),
                torchvision.transforms.Lambda(lambda data: data.float()),
            ]
        )
        self.target_transform = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                transform_lib.Lambda(lambda x: (255 * x).int()),
                transform_lib.Lambda(lambda x: x.apply_(map_class_to_zero_dict.get)),
                torchvision.transforms.Resize(self.hparams.resize, antialias=True),
                torchvision.transforms.Lambda(lambda data: torch.bucketize(data, boundaries)),
                torchvision.transforms.Lambda(lambda data: torch.nn.functional.one_hot(data, num_classes=num_states)),
                torchvision.transforms.Lambda(lambda data: einops.rearrange(data, "c h w s -> s h w c").squeeze(-1)),
                torchvision.transforms.Lambda(lambda data: data.float()),
            ]
        )

        self.num_workers = 4 * torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"Using {self.num_workers=} in CityscapesDataModule")

    def prepare_data(self):
        # download
        Cityscapes(root=self.data_dir, split="train", mode="coarse", target_type="semantic")
        Cityscapes(root=self.data_dir, split="val", mode="coarse", target_type="semantic")
        Cityscapes(root=self.data_dir, split="train", mode="fine", target_type="semantic")
        Cityscapes(root=self.data_dir, split="val", mode="fine", target_type="semantic")

    def label_filter(self, dataset, labels):
        mask = torch.isin(dataset.targets, torch.Tensor(labels))
        dataset.data = dataset.data[mask]
        dataset.targets = dataset.targets[mask]

    def setup(self, stage: str = "fit"):
        # Assign train/val datasets for use in dataloaders

        if stage == "fit":
            self.train_dataset = Cityscapes(root=self.data_dir, split="train", mode="fine", target_type="semantic",
                                            transform=self.transform, target_transform=self.target_transform)
            self.val_dataset = Cityscapes(root=self.data_dir, split="val", mode="fine", target_type="semantic",
                                          transform=self.transform, target_transform=self.target_transform)

        # img, target = self.train_dataset[0]
        # print(f"{img.shape=} {target.shape=}")

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = Cityscapes(root=self.data_dir, split="val", mode="coarse", target_type="semantic")

        if stage == "predict":
            self.predict_dataset = Cityscapes(root=self.data_dir, split="val", mode="coarse", target_type="semantic")

    def train_dataloader(self, batch_size=None):
        return DataLoader(self.train_dataset, batch_size=batch_size or self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self, batch_size=None):
        return DataLoader(self.val_dataset, batch_size=batch_size or self.batch_size, num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self, batch_size=None):
        return DataLoader(self.test_dataset, batch_size=batch_size or self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    dm = CityscapesDataModule(num_states=-1, data_dir=path / "data/cityscapes", resize=[100, 200])
    dm.setup()
    dm.prepare_data()
    for batch in dm.val_dataloader(batch_size=3):
        target = batch
        for target_ in target:
            # print(f"{img_.shape=}")
            fig, axs = plt.subplots(1, 2)
            axs = axs.flatten()
            # axs[0].imshow(img_.permute(1, 2, 0))
            # target = target.float()/ 20.
            # print(f"{target.unique()=}")
            # target[target >= 10] = 0
            axs[1].matshow(target_.squeeze())
            plt.show()
        break
