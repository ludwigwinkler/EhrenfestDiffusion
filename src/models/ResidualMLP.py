# System 
import os, sys
from pathlib import Path

from torchtyping import TensorType, patch_typeguard

# Python
import numpy as np
import matplotlib.pyplot as plt
import functools

# Bookkeeping
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from simple_parsing.helpers import JsonSerializable
import pprint
from typing import *

# Machine Learning
import torch, jax, einops, math
from torch import nn
from torch import functional as F
import jax.numpy as jnp
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer


def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1, f"{timesteps.shape=}"  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


class NiN(nn.Module):
    def __init__(self, in_ch, out_ch, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_ch, out_ch)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(out_ch), requires_grad=True)

    def forward(self, x: TensorType["batch", "in_ch", "H", "W"]
                ) -> TensorType["batch", "out_ch", "H", "W"]:
        x = x.permute(0, 2, 3, 1)
        # x (batch, H, W, in_ch)
        y = torch.einsum('bhwi,ik->bhwk', x, self.W) + self.b
        # y (batch, H, W, out_ch)
        return y.permute(0, 3, 1, 2)


class AttnBlock(nn.Module):
    """Channel-wise self-attention block."""

    def __init__(self, channels, skip_rescale=True):
        super().__init__()
        self.skip_rescale = skip_rescale
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32),
                                        num_channels=channels, eps=1e-6)
        self.NIN_0 = NiN(channels, channels)
        self.NIN_1 = NiN(channels, channels)
        self.NIN_2 = NiN(channels, channels)
        self.NIN_3 = NiN(channels, channels, init_scale=0.)

    def forward(self, x: TensorType["batch", "channels", "H", "W"]
                ) -> TensorType["batch", "channels", "H", "W"]:

        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)

        if self.skip_rescale:
            return (x + h) / np.sqrt(2.)
        else:
            return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim=None, dropout=0.1, skip_rescale=True):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.skip_rescale = skip_rescale

        self.act = nn.functional.silu
        self.groupnorm0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32),
            num_channels=in_ch, eps=1e-6
        )
        self.conv0 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=1
        )

        if temb_dim is not None:
            self.dense0 = nn.Linear(temb_dim, out_ch)
            nn.init.zeros_(self.dense0.bias)

        self.groupnorm1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32),
            num_channels=out_ch, eps=1e-6
        )
        self.dropout0 = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1
        )
        if out_ch != in_ch:
            self.nin = NiN(in_ch, out_ch)

    def forward(self, x: TensorType["batch", "in_ch", "H", "W"],
                temb: TensorType["batch", "temb_dim"] = None
                ) -> TensorType["batch", "out_ch", "H", "W"]:

        assert x.shape[1] == self.in_ch

        h = self.groupnorm0(x)
        h = self.act(h)
        h = self.conv0(h)

        if temb is not None:
            h += self.dense0(self.act(temb))[:, :, None, None]

        h = self.groupnorm1(h)
        h = self.act(h)
        h = self.dropout0(h)
        h = self.conv1(h)
        if h.shape[1] != self.in_ch:
            x = self.nin(x)

        assert x.shape == h.shape

        if self.skip_rescale:
            return (x + h) / np.sqrt(2.)
        else:
            return x + h


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3,
                              stride=2, padding=0)

    def forward(self, x: TensorType["batch", "ch", "inH", "inW"]
                ) -> TensorType["batch", "ch", "outH", "outW"]:
        B, C, H, W = x.shape
        x = nn.functional.pad(x, (0, 1, 0, 1))
        x = self.conv(x)

        assert x.shape == (B, C, H // 2, W // 2)
        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: TensorType["batch", "ch", "inH", "inW"]
                ) -> TensorType["batch", "ch", "outH", "outW"]:
        B, C, H, W = x.shape
        h = F.interpolate(x, (H * 2, W * 2), mode='nearest')
        h = self.conv(h)

        assert h.shape == (B, C, H * 2, W * 2)
        return h


class UNet(nn.Module):
    def __init__(self, ch, num_res_blocks, num_scales, ch_mult, input_channels,
                 output_channels, scale_count_to_put_attn, data_min_max, dropout,
                 skip_rescale, do_time_embed, time_scale_factor=None, time_embed_dim=None):
        super().__init__()
        assert num_scales == len(ch_mult)

        self.ch = ch
        self.num_res_blocks = num_res_blocks
        self.num_scales = num_scales
        self.ch_mult = ch_mult
        self.input_channels = input_channels
        self.output_channels = 2 * input_channels
        self.scale_count_to_put_attn = scale_count_to_put_attn
        self.data_min_max = data_min_max  # tuple of min and max value of input so it can be rescaled to [-1, 1]
        self.dropout = dropout
        self.skip_rescale = skip_rescale
        self.do_time_embed = do_time_embed  # Whether to add in time embeddings
        self.time_scale_factor = time_scale_factor  # scale to make the range of times be 0 to 1000
        self.time_embed_dim = time_embed_dim

        self.act = nn.functional.silu

        if self.do_time_embed:
            self.temb_modules = []
            self.temb_modules.append(nn.Linear(self.time_embed_dim, self.time_embed_dim * 4))
            nn.init.zeros_(self.temb_modules[-1].bias)
            self.temb_modules.append(nn.Linear(self.time_embed_dim * 4, self.time_embed_dim * 4))
            nn.init.zeros_(self.temb_modules[-1].bias)
            self.temb_modules = nn.ModuleList(self.temb_modules)

        self.expanded_time_dim = 4 * self.time_embed_dim if self.do_time_embed else None

        self.input_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=self.ch,
            kernel_size=3, padding=1
        )

        h_cs = [self.ch]
        in_ch = self.ch

        # Downsampling
        self.downsampling_modules = []

        for scale_count in range(self.num_scales):
            for res_count in range(self.num_res_blocks):
                out_ch = self.ch * self.ch_mult[scale_count]
                self.downsampling_modules.append(
                    ResBlock(in_ch, out_ch, temb_dim=self.expanded_time_dim,
                             dropout=dropout, skip_rescale=self.skip_rescale)
                )
                in_ch = out_ch
                h_cs.append(in_ch)
                if scale_count == self.scale_count_to_put_attn:
                    self.downsampling_modules.append(
                        AttnBlock(in_ch, skip_rescale=self.skip_rescale)
                    )

            if scale_count != self.num_scales - 1:
                self.downsampling_modules.append(Downsample(in_ch))
                h_cs.append(in_ch)

        self.downsampling_modules = nn.ModuleList(self.downsampling_modules)

        # Middle
        self.middle_modules = []

        self.middle_modules.append(
            ResBlock(in_ch, in_ch, temb_dim=self.expanded_time_dim,
                     dropout=dropout, skip_rescale=self.skip_rescale)
        )
        self.middle_modules.append(
            AttnBlock(in_ch, skip_rescale=self.skip_rescale)
        )
        self.middle_modules.append(
            ResBlock(in_ch, in_ch, temb_dim=self.expanded_time_dim,
                     dropout=dropout, skip_rescale=self.skip_rescale)
        )
        self.middle_modules = nn.ModuleList(self.middle_modules)

        # Upsampling
        self.upsampling_modules = []

        for scale_count in reversed(range(self.num_scales)):
            for res_count in range(self.num_res_blocks + 1):
                out_ch = self.ch * self.ch_mult[scale_count]
                self.upsampling_modules.append(
                    ResBlock(in_ch + h_cs.pop(),
                             out_ch,
                             temb_dim=self.expanded_time_dim,
                             dropout=dropout,
                             skip_rescale=self.skip_rescale
                             )
                )
                in_ch = out_ch

                if scale_count == self.scale_count_to_put_attn:
                    self.upsampling_modules.append(
                        AttnBlock(in_ch, skip_rescale=self.skip_rescale)
                    )
            if scale_count != 0:
                self.upsampling_modules.append(Upsample(in_ch))

        self.upsampling_modules = nn.ModuleList(self.upsampling_modules)

        assert len(h_cs) == 0

        # output
        self.output_modules = []

        self.output_modules.append(
            nn.GroupNorm(min(in_ch // 4, 32), in_ch, eps=1e-6)
        )

        self.output_modules.append(
            nn.Conv2d(in_ch, self.output_channels, kernel_size=3, padding=1)
        )
        self.output_modules = nn.ModuleList(self.output_modules)

    def _center_data(self, x):
        out = (x - self.data_min_max[0]) / (self.data_min_max[1] - self.data_min_max[0])  # [0, 1]
        return 2 * out - 1  # to put it in [-1, 1]

    def _time_embedding(self, timesteps):
        if self.do_time_embed:
            temb = transformer_timestep_embedding(
                timesteps * self.time_scale_factor, self.time_embed_dim
            )
            temb = self.temb_modules[0](temb)
            temb = self.temb_modules[1](self.act(temb))
        else:
            temb = None

        return temb

    def _do_input_conv(self, h):
        h = self.input_conv(h)
        hs = [h]
        return h, hs

    def _do_downsampling(self, h, hs, temb):
        m_idx = 0
        for scale_count in range(self.num_scales):
            for res_count in range(self.num_res_blocks):
                h = self.downsampling_modules[m_idx](h, temb)
                m_idx += 1
                if scale_count == self.scale_count_to_put_attn:
                    h = self.downsampling_modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            if scale_count != self.num_scales - 1:
                h = self.downsampling_modules[m_idx](h)
                hs.append(h)
                m_idx += 1

        assert m_idx == len(self.downsampling_modules)

        return h, hs

    def _do_middle(self, h, temb):
        m_idx = 0
        h = self.middle_modules[m_idx](h, temb)
        m_idx += 1
        h = self.middle_modules[m_idx](h)
        m_idx += 1
        h = self.middle_modules[m_idx](h, temb)
        m_idx += 1

        assert m_idx == len(self.middle_modules)

        return h

    def _do_upsampling(self, h, hs, temb):
        m_idx = 0
        for scale_count in reversed(range(self.num_scales)):
            for res_count in range(self.num_res_blocks + 1):
                h = self.upsampling_modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

                if scale_count == self.scale_count_to_put_attn:
                    h = self.upsampling_modules[m_idx](h)
                    m_idx += 1

            if scale_count != 0:
                h = self.upsampling_modules[m_idx](h)
                m_idx += 1

        assert len(hs) == 0
        assert m_idx == len(self.upsampling_modules)

        return h

    def _do_output(self, h):

        h = self.output_modules[0](h)
        h = self.act(h)
        h = self.output_modules[1](h)

        return h

    def _logistic_output_res(self,
                             h: TensorType["B", "twoC", "H", "W"],
                             centered_x_in: TensorType["B", "C", "H", "W"]
                             ) -> TensorType["B", "twoC", "H", "W"]:
        B, twoC, H, W = h.shape
        C = twoC // 2
        h[:, 0:C, :, :] = torch.tanh(centered_x_in + h[:, 0:C, :, :])
        return h

    def forward(self,
                x: TensorType["B", "C", "H", "W"],
                timesteps: TensorType["B"] = None
                ) -> TensorType["B", "twoC", "H", "W"]:

        h = self._center_data(x)
        centered_x_in = h

        temb = self._time_embedding(timesteps)

        h, hs = self._do_input_conv(h)

        h, hs = self._do_downsampling(h, hs, temb)

        h = self._do_middle(h, temb)

        h = self._do_upsampling(h, hs, temb)

        h = self._do_output(h)

        # h (B, 2*C, H, W)
        h = self._logistic_output_res(h, centered_x_in)

        return h


# Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, device, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x: TensorType["B", "L", "K"]
                ) -> TensorType["B", "L", "K"]:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, 0:x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, temb_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads,
                                               dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.film_from_temb = nn.Linear(temb_dim, 2 * d_model)

    def forward(self,
                x: TensorType["B", "L", "K"],
                temb: TensorType["B", "temb_dim"]
                ):
        B, L, K = x.shape

        film_params = self.film_from_temb(temb)

        x = self.norm1(x + self._sa_block(x))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]
        x = self.norm2(x + self._ff_block(x))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]

        return x

    def _sa_block(self, x):
        x = self.self_attn(x, x, x)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FFResidual(nn.Module):
    def __init__(self, d_model, hidden, temb_dim):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.activation = nn.ReLU()

        self.film_from_temb = nn.Linear(temb_dim, 2 * d_model)

    def forward(self, x, temb):
        B, L, K = x.shape

        film_params = self.film_from_temb(temb)

        x = self.norm(x + self.linear2(self.activation(self.linear1(x))))
        x = film_params[:, None, 0:K] * x + film_params[:, None, K:]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward,
                 dropout, num_output_FFresiduals, time_scale_factor, S, max_len,
                 temb_dim, use_one_hot_input, device):
        super().__init__()

        self.temb_dim = temb_dim
        self.use_one_hot_input = use_one_hot_input

        self.S = S

        self.pos_embed = PositionalEncoding(device, d_model, dropout, max_len)

        self.encoder_layers = []
        for i in range(num_layers):
            self.encoder_layers.append(
                TransformerEncoderLayer(d_model, num_heads, dim_feedforward,
                                        dropout, 4 * temb_dim)
            )
        self.encoder_layers = nn.ModuleList(self.encoder_layers)

        self.output_resid_layers = []
        for i in range(num_output_FFresiduals):
            self.output_resid_layers.append(
                FFResidual(d_model, dim_feedforward, 4 * temb_dim)
            )
        self.output_resid_layers = nn.ModuleList(self.output_resid_layers)

        self.output_linear = nn.Linear(d_model, self.S)

        if use_one_hot_input:
            self.input_embedding = nn.Linear(S, d_model)
        else:
            self.input_embedding = nn.Linear(1, d_model)

        self.temb_net = nn.Sequential(
            nn.Linear(temb_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 4 * temb_dim)
        )

        self.time_scale_factor = time_scale_factor

    def forward(self, x: TensorType["B", "L"],
                times: TensorType["B"]):
        B, L = x.shape

        temb = self.temb_net(
            transformer_timestep_embedding(
                times * self.time_scale_factor, self.temb_dim
            )
        )
        one_hot_x = nn.functional.one_hot(x, num_classes=self.S)  # (B, L, S)

        if self.use_one_hot_input:
            x = self.input_embedding(one_hot_x.float())  # (B, L, K)
        else:
            x = self.normalize_input(x)
            x = x.view(B, L, 1)
            x = self.input_embedding(x)  # (B, L, K)

        x = self.pos_embed(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, temb)

        # x (B, L, K)
        for resid_layer in self.output_resid_layers:
            x = resid_layer(x, temb)

        x = self.output_linear(x)  # (B, L, S)

        x = x + one_hot_x

        return x

    def normalize_input(self, x):
        x = x / self.S  # (0, 1)
        x = x * 2 - 1  # (-1, 1)
        return x


class ResidualMLP(nn.Module):
    def __init__(self, num_layers, d_model, hidden_dim, D,
                 time_scale_factor, temb_dim, out_dim):
        super().__init__()

        self.time_scale_factor = time_scale_factor
        self.d_model = d_model
        self.num_layers = num_layers
        self.temb_dim = temb_dim

        self.activation = nn.ReLU()

        self.input_layer = nn.Linear(D, d_model)

        self.layers1 = []
        self.layers2 = []
        self.norm_layers = []
        self.temb_layers = []
        for n in range(num_layers):
            self.layers1.append(
                nn.Linear(d_model, hidden_dim)
            )
            self.layers2.append(
                nn.Linear(hidden_dim, d_model)
            )
            self.norm_layers.append(
                nn.LayerNorm(d_model)
            )
            self.temb_layers.append(
                nn.Linear(4 * temb_dim, 2 * d_model)
            )

        self.layers1 = nn.ModuleList(self.layers1)
        self.layers2 = nn.ModuleList(self.layers2)
        self.norm_layers = nn.ModuleList(self.norm_layers)
        self.temb_layers = nn.ModuleList(self.temb_layers)

        self.output_layer = nn.Linear(d_model, out_dim)

        self.temb_net = nn.Sequential(
            nn.Linear(temb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4 * temb_dim)
        )

    def forward(self,
                x: TensorType["B", "D"],
                times: TensorType["B"]
                ):
        # B, D = x.shape
        # S = self.S

        temb = self.temb_net(
            transformer_timestep_embedding(
                times * self.time_scale_factor, self.temb_dim
            )
        )

        # one_hot_x = nn.functional.one_hot(x, num_classes=self.S)  # (B, D, S)

        # h = self.normalize_input(x)
        h = x
        h = self.input_layer(h)  # (B, d_model)

        for n in range(self.num_layers):
            h = self.norm_layers[n](h + self.layers2[n](self.activation(self.layers1[n](h))))
            film_params = self.temb_layers[n](temb)
            h = film_params[:, 0:self.d_model] * h + film_params[:, self.d_model:]

        h = self.output_layer(h)  # (B, D*S)

        # h = h.reshape(B, D, S)

        # logits = h + one_hot_x

        return h

    # def normalize_input(self, x):
    #     x = x / self.S  # (0, 1)
    #     x = x * 2 - 1  # (-1, 1)
    #     return x


class ResidualMLPModel(torch.nn.Module):
    def __init__(self, cfg, out_dim, output_type):
        super().__init__()

        num_layers = 5
        d_model = 124
        hidden_dim = 124
        time_scale_factor = 1000
        temb_dim = 124

        # assert len(cfg.data.shape) == 1

        tmp_net = ResidualMLP(
            num_layers, d_model, hidden_dim, 2,
            time_scale_factor, temb_dim, out_dim
        )

        self.net = tmp_net

        # self.data_shape = cfg.data.shape
        self.output_type = output_type

    def forward(self,
                x: TensorType["B", "D"],
                t: TensorType["B"]
                ) -> TensorType["B", "D", "S"]:
        """
            Returns logits over state space
        """

        output = self.net(x, t)  # (B, D, S)
        if self.output_type == 'score_x0':  # predict x0 in score term
            output = output  # little inductive bias
            output_dict = {'score_x0': output, 'x_0_t': output}
        elif self.output_type == 'score':
            output_dict = {'score': output, 'x_0_t': output}
        elif self.output_type == 'x0':
            output_dict = {'x0': output, 'x_0_t': output, 'output': output}
        elif self.output_type == 'taylor1':
            output_dict = {'taylor1': output, 'x_0_t': output, 'output': output}
        elif self.output_type == "gaussian":
            pred_mu, pred_logscale = output.chunk(chunks=2, dim=1)
            pred_mu = pred_mu
            pred_scale = torch.nn.functional.softplus(pred_logscale)
            output_dict = {'gaussian_mu': pred_mu, 'gaussian_scale': pred_scale, 'x_0_t': pred_mu}
        elif self.output_type == "ratio":
            death_rate, birth_rate = output.chunk(chunks=2, dim=1)
            ratio = output
            output_dict = {'death_rate': death_rate, 'birth_rate': birth_rate, 'ratio': ratio, 'x_0_t': x,
                           'output': output}

        return output_dict

    def prob_x0_xt(self, x, t, S, pred: dict = None):

        if pred is None:
            pred = self.forward(x, t)

        if "gaussian_mu" in pred and 'gaussian_scale' in pred:
            pred_mu, pred_scale = pred["gaussian_mu"], pred["gaussian_scale"]
            bin_width = 2. / S
            s = torch.linspace(start=-1. + bin_width / 2, end=1. - bin_width / 2, steps=S,
                               device=pred_mu.device, dtype=pred_mu.dtype)  # S=256 -> [-127, 128]
            x0_s = torch.ones(pred_mu.shape + (s.numel(),), device=pred_mu.device, dtype=pred_mu.dtype) * s
            '''p(x_0 | x_t) under variational distribution'''
            pred_ = torch.distributions.Normal(loc=pred_mu.unsqueeze(-1), scale=pred_scale.unsqueeze(-1),
                                               validate_args=False)
            prob_x0 = pred_.log_prob(x0_s).exp()  # [BS, C, H, W, S]
            prob_x0 = prob_x0 / prob_x0.sum(dim=-1, keepdim=True)
            pred['prob'] = prob_x0
            return pred

        elif "logistic_mu" in pred and "logistic_scale" in pred:
            mu, log_scale = pred['logistic_mu'], pred[
                'logistic_scale']  # tuple of (torch.Tensor: [BS, C, H, W], torch.Tensor: [BS, C, H, W])
            mu, log_scale = mu.unsqueeze(-1), log_scale.unsqueeze(-1)
            inv_scale = torch.exp(- (log_scale - 2))

            _log_minus_exp = lambda a, b: a + torch.log1p(-torch.exp(b - a) + 1e-6)

            bin_width = 2. / S
            bin_centers = torch.linspace(start=-1. + bin_width / 2,
                                         end=1. - bin_width / 2,
                                         steps=S,
                                         device=mu.device).view(1, 1, 1, 1, S)

            sig_in_left = (bin_centers - bin_width / 2 - mu) * inv_scale
            bin_left_logcdf = F.logsigmoid(sig_in_left)
            sig_in_right = (bin_centers + bin_width / 2 - mu) * inv_scale
            bin_right_logcdf = F.logsigmoid(sig_in_right)

            logits_1 = _log_minus_exp(bin_right_logcdf, bin_left_logcdf)
            logits_2 = _log_minus_exp(-sig_in_left + bin_left_logcdf, -sig_in_right + bin_right_logcdf)
            if False:
                logits = torch.min(logits_1, logits_2)
            else:
                logits = logits_1
            pred['prob'] = F.softmax(logits, dim=-1)
            return pred
        else:
            return pred
