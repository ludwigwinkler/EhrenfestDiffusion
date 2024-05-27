from .Layers import (  # ResidualBlock,
    ConditionalResidualBlock,
    CondRefineBlock,
    RefineBlock,
    get_act,
    ncsn_conv3x3,
)

CondResidualBlock = ConditionalResidualBlock
conv3x3 = ncsn_conv3x3

from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import einsum
from torch.nn import Conv2d, GroupNorm, LayerNorm, Linear, Module, ModuleList


class ConditionNHWC(Module):
    def __init__(self, out_features):
        super().__init__()
        inv_freq = 1.0 / torch.logspace(-5, 5, out_features // 2)
        self.inv_freq = torch.nn.Parameter(inv_freq, requires_grad=False)

    # self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, condition):
        freqs = torch.outer(condition, self.inv_freq)  # c = d / 2
        posemb = repeat(freqs, "b c -> b (2 c)")
        odds, evens = rearrange(x, "... (j c) -> ... j c", j=2).unbind(dim=-2)
        rotated = torch.cat((-evens, odds), dim=-1)
        # plt.plot(posemb.cos(), alpha=0.5)
        # plt.show()
        # plt.plot(posemb.sin(), alpha=0.5)
        # plt.show()
        return einsum("b ... d , b ... d -> b ... d", x, posemb.cos()) + einsum("b ... d , b ... d -> b ... d", rotated, posemb.sin())


class ConditionNCHW(Module):
    def __init__(self, out_features):
        super().__init__()
        inv_freq = 1.0 / torch.logspace(-5, 5, out_features // 2)
        self.inv_freq = torch.nn.Parameter(inv_freq, requires_grad=False)

        if False:
            print("Visualizing embeddings ...")
            t = torch.linspace(0, 10, 100)
            freqs = torch.outer(t, self.inv_freq)  # c = d / 2
            posemb = repeat(freqs, "b c -> b (2 c)")
            print(f"{posemb.cos().shape=} {posemb.sin().shape=}")
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))
            axs[0].plot(t, posemb.cos()[:, -20:], alpha=0.2)
            axs[1].plot(t, posemb.sin()[:, -20:], alpha=0.2)
            axs[2].plot(t, posemb.sin()[:, -20:] + posemb.cos()[:, -20:], alpha=0.2)
            plt.suptitle("All 4th frequencies ([::4])")
            plt.show()
            exit()

    def forward(self, x, condition):
        """

        :param x: data
        :param condition: time
        :return:
        """
        time_offset = 0.0
        freqs = torch.outer(condition + time_offset, self.inv_freq)  # c = d / 2
        posemb = repeat(freqs, "b c -> b (2 c)")
        odds, evens = rearrange(x, "b (j c) ... -> b j c ...", j=2).unbind(dim=1)
        rotated = torch.cat((-evens, odds), dim=1)
        return einsum("b d ... , b d ... -> b d ...", x, posemb.cos()) + einsum("b d ... , b d ... -> b d ...", rotated, posemb.sin())


class SelfAttention(Module):
    def __init__(self, head_dim: int, heads: int):
        super().__init__()
        hidden_dim = head_dim * heads
        self.head_dim = head_dim
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.in_proj = Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        b, h, w, d = x.shape
        x = rearrange(x, "b h w d -> b (h w) d")
        p = self.in_proj(x)
        q, k, v = torch.split(
            p,
            [
                self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim,
            ],
            -1,
        )
        (q, k, v) = map(lambda x: rearrange(x, "b i (h d) -> b i h d", h=self.heads), (q, k, v))
        a = einsum("b i h d, b j h d -> b h i j", q, k) * (self.head_dim**-0.5)
        a = F.softmax(a, dim=-1)
        o = einsum("b h i j, b j h d -> b i h d", a, v)
        o = rearrange(o, "b i h d -> b i (h d)")
        x = self.out_proj(o)
        x = rearrange(x, "b (h w) d -> b h w d", h=h, w=w)
        return x


class ConditionedSequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = ModuleList(layers)

    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.condition = ConditionNCHW(out_channels) if out_channels % 2 == 0 else None
        self.layers = ModuleList(
            [
                Conv2d(in_channels, out_channels, (1, 1)),
                Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1),
                Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1),
            ]
        )
        self.norm = GroupNorm(1, out_channels)

    def forward(self, x, condition):
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
            else:
                if self.condition:
                    x = x + layer(self.condition(F.gelu(self.norm(x)), condition=condition))
                else:
                    x = x + layer(F.gelu(self.norm(x)))
        return x


class BottleneckBlock(Module):
    def __init__(self, channels):
        super().__init__()
        self.condition = ConditionNHWC(channels)
        self.layers = ModuleList([SelfAttention(channels // 8, 8) for _ in range(4)])
        self.norm = LayerNorm(channels)

    def forward(self, x, condition):
        x = rearrange(x, "b c h w -> b h w c")
        for layer in self.layers:
            x = x + layer(self.condition(self.norm(x), condition=condition))
        x = rearrange(x, "b h w c -> b c h w")
        return x


class Bicubic(Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x, *args, **kwargs):
        return F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear")


class SmallUNetConstructor(Module):
    def __init__(self, encoders_decoders: Sequence[Tuple[Module, Module]], bottleneck: Module):
        super().__init__()
        outer_pair, *inner_remaining = encoders_decoders
        self.encoder, self.decoder = outer_pair
        if inner_remaining:
            self.bottleneck = SmallUNetConstructor(inner_remaining, bottleneck)
        else:
            self.bottleneck = bottleneck

    def forward(self, x, condition):
        encoded = self.encoder(x, condition=condition)
        bottlenecked = self.bottleneck(encoded, condition=condition)
        return self.decoder(torch.cat([encoded, bottlenecked], dim=1), condition=condition)


class SmallUNet(Module):
    def __init__(self, in_channels, mid_channels=128, out_channels=3, residual=False, output_type="birthdeath"):
        super().__init__()
        self.output_type = output_type
        self.residual = residual

        # self.net = SmallUNetConstructor(
        #     [
        #         (ResidualBlock(in_channels, 64), ResidualBlock(64 + 64, out_channels)),
        #         (ConditionedSequential(Bicubic(1 / 2), ResidualBlock(64, 128)), ConditionedSequential(ResidualBlock(128 + 128, 64), Bicubic(2))),
        #         (ConditionedSequential(Bicubic(1 / 2), ResidualBlock(128, 256)), ConditionedSequential(ResidualBlock(256 + 256, 128), Bicubic(2))),
        #         (ConditionedSequential(Bicubic(1 / 2), ResidualBlock(256, 512)), ConditionedSequential(ResidualBlock(512 + 512, 256), Bicubic(2))),
        #     ],
        #     BottleneckBlock(512),
        # )
        k = 2
        f = 32
        self.net = SmallUNetConstructor(
            [
                (ResidualBlock(in_channels, k * f), ResidualBlock(2 * k * f, out_channels)),
                (ConditionedSequential(Bicubic(1 / 2), ResidualBlock(k * f, k * 2 * f)), ConditionedSequential(ResidualBlock(2 * k * 2 * f, k * f), Bicubic(2))),
                (ConditionedSequential(Bicubic(1 / 2), ResidualBlock(k * 2 * f, k * 4 * f)), ConditionedSequential(ResidualBlock(2 * k * 4 * f, k * 2 * f), Bicubic(2))),
                (ConditionedSequential(Bicubic(1 / 2), ResidualBlock(k * 4 * f, k * 8 * f)), ConditionedSequential(ResidualBlock(k * 16 * f, k * 4 * f), Bicubic(2))),
            ],
            # BottleneckBlock(1024),
            BottleneckBlock(k * 8 * f),
        )

    def forward(self, x, t):
        if self.output_type == "birthdeath":
            assert x.min() >= -1.0 and x.max() <= 1.0
        # print(f"{x.shape=}")
        # exit('smallunet.forward')
        pred = self.net(x, condition=t)
        if self.output_type == "birthdeath":
            out = torch.tanh(x + pred) if self.residual else torch.tanh(pred)
        elif self.output_type == "variational":
            pred_mu, pred_logscale = pred.chunk(chunks=2, dim=1)
            pred_mu = x + pred_mu if self.residual else pred_mu
            pred_scale = torch.nn.functional.softplus(pred_logscale)
            # pred_scale = torch.ones_like(pred_scale).fill_(0.1)
            out = torch.distributions.Normal(loc=pred_mu, scale=pred_scale)
        elif self.output_type == "categorical":
            assert pred.shape[1] == x.shape[1], f"DiffuserNet: {x.shape} vs {pred.shape}"
            out = pred
        return out


if __name__ == "__main__":
    emb = ConditionNHWC(out_features=60)
    print(emb(torch.randn(200, 60), torch.linspace(0, 10, 200)).shape)
