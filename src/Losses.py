import functools as ft
import sys
import einops
import matplotlib.pyplot as plt
import torch
import torchvision.utils
from einops import rearrange
from torch.distributions import Categorical
from torch.nn.functional import one_hot
from torch.distributions import Normal, Categorical
import seaborn as sns

from typing import Tuple

from .NoiseSchedules import OrdinalNoiseSchedule


class OrdinalRegression(torch.nn.Module):
    def __init__(self, criterion_config, model_config):
        super().__init__()
        self.criterion_config = criterion_config
        self.model_config = model_config

    def forward(self, model, x0, t, noise_schedule):
        # print(type(noise_schedule))
        assert isinstance(noise_schedule,
                          OrdinalNoiseSchedule), f"{noise_schedule.__class__.__mro__=} \n {type(noise_schedule)=}"
        p_t = noise_schedule.p_t(t=t, x0=x0)
        S = noise_schedule.num_states - 1
        x_t = Categorical(p_t.float()).sample()  # [BS, C, H, W, S] -> [BS, C, H, W]

        norm_fn = lambda x: 2.0 * x / S - 1.0
        x_t_norm = norm_fn(x_t)
        x0_norm = norm_fn(x0)

        assert x_t_norm.min() >= -1.0 and x_t_norm.max() <= 1.0

        pred_x0_norm = model(x=x_t_norm, t=t)  # [-1,1]
        pred = (pred_x0_norm + 1.0) * S / 2.0  # rescale back in original space
        assert pred_x0_norm.shape == x0_norm.shape

        loss = torch.nn.functional.mse_loss(input=pred_x0_norm, target=x0_norm)

        return {"loss": loss, "p_t": p_t, "pred": pred, "target": x0_norm, "x_0": x0, "x_t_norm": x_t, "t": t}


class EhrenfestRegression(torch.nn.Module):
    def __init__(self, criterion_config, model_config):
        super().__init__()

        self.criterion_config = criterion_config
        self.model_config = model_config

    def forward(self, model, x0, t, noise_schedule):
        # print(type(noise_schedule))
        assert isinstance(noise_schedule,
                          OrdinalNoiseSchedule), f"{noise_schedule.__class__.__mro__=} \n {type(noise_schedule)=}"
        # assert -noise_schedule.num_states / 2 <= x0.min() and x0.max() <= noise_schedule.num_states / 2, f"{x0.min()=} {x0.max()=} {noise_schedule.num_states=}"
        _, mu, std = noise_schedule.compute_moments(t=t, x0=x0)

        '''Setting up required DDPM Quantities'''
        _, mu, std = noise_schedule.compute_moments(t=t, x0=x0)  # OU solution
        var = std.pow(2)
        w_t = noise_schedule.w_t(t)
        epsilon = torch.randn_like(mu)
        x_t = mu + epsilon * std

        '''Computing Loss: DDPM predicts noise towards mode of OU process'''
        output_dict = model(x=x_t, t=t)  # ([K C H W], [K]) -> {'key': [K C H W], ... }

        criterion_dict = {'output': output_dict['output'], "x_0": x0, "x_t": x_t, "t": t}
        '''Reconstructing x0 from x_t and predicted ϵ_θ(x_t)'''
        if model.output_type in ['epsilon']:
            x0_t = (x_t - std * output_dict['epsilon'].detach()) / w_t.view(-1, 1, 1, 1)
            loss = ((output_dict['epsilon'] - epsilon)).pow(2).mean()
            criterion_dict['loss'] = loss
            criterion_dict['target'] = epsilon
            criterion_dict['x_0_t'] = x0_t
            criterion_dict['epsilon'] = output_dict['epsilon'].detach()

        elif model.output_type in ['score'] and 'score' in output_dict:
            '''
            Score: ∇_x log N(x ; mu, std) = - (x - w(t) x_0) / std^2
            We train on z-score -(x - mu) / std
            '''
            target = - (x_t - mu) / std.pow(2)
            loss = ((output_dict['score'] - target)).pow(2).mean()
            criterion_dict['x_0_t'] = x_t
            criterion_dict['loss'] = loss
            criterion_dict['target'] = target
        elif model.output_type in ['taylor1'] and 'taylor1' in output_dict:
            '''
            Taylor1: ∇_x log N(x ; mu, std) = - (x - w(t) x_0) / std^2
            We train on z-score -(x - mu) / std
            '''
            target = mu
            loss = ((output_dict['taylor1'] - target)).pow(2).mean()
            criterion_dict['x_0_t'] = x_t
            criterion_dict['loss'] = loss
            criterion_dict['target'] = target
        elif model.output_type in ['ratio']:
            '''
            x_t: [K C H W] -> [K [C x H x W]] -> [K D]
            mu_t_x0: [K C H W]
            var_t_x0: [K C H W]
            birth_ratio: [K C H W] 
            '''
            scaling = 2 / 256
            birth_ratio = torch.exp((- 2 * (x_t - mu) * scaling - scaling ** 2) / (2 * var))
            death_ratio = torch.exp((+ 2 * (x_t - mu) * scaling - scaling ** 2) / (2 * var))
            ratio = torch.concat([death_ratio, birth_ratio], dim=1)  # [K, 2C, H, W]


            target = ratio.clamp(0, 3)  # from ratio histograms, this is a sensible range
            assert output_dict['ratio'].shape == ratio.shape
            loss = ((output_dict['ratio'] - target)).clamp(-10, 10).pow(
                2).mean()  # ([K, 2C, H, W] - [K, 2C, H, W]).pow(2).mean()
            criterion_dict['x_0_t'] = x_t
            criterion_dict['loss'] = loss
            criterion_dict['target'] = epsilon
        elif model.output_type in ['ratio2']:
            '''
            x_t: [K C H W] -> [K [C x H x W]] -> [K D]
            mu_t_x0: [K C H W]
            var_t_x0: [K C H W]
            birth_ratio: [K C H W] 
            '''
            scaling = 2 / 256
            birth_ratio = torch.exp(
                (- 2 * (x_t - mu) * scaling - scaling ** 2) / (2 * var))  # exploding ratio prevention
            death_ratio = torch.exp((+ 2 * (x_t - mu) * scaling - scaling ** 2) / (2 * var))
            ratio = torch.concat([death_ratio, birth_ratio], dim=1)  # [K, 2C, H, W]

            target = ratio.clamp(0, 3)  # from ratio histograms, this is a sensible range
            assert output_dict['ratio2'].shape == ratio.shape, f"{output_dict['ratio2'].shape=} {ratio.shape=}"
            loss = ((output_dict['ratio2'] - target)).clamp(-10, 10).pow(
                2).mean()  # ([K, 2C, H, W] - [K, 2C, H, W]).pow(2).mean()
            criterion_dict['x_0_t'] = x_t
            criterion_dict['loss'] = loss
            criterion_dict['target'] = target


        return criterion_dict


class EhrenfestVariational(torch.nn.Module):
    def __init__(self, criterion_config, model_config):
        super().__init__()

        self.criterion_config = criterion_config
        self.model_config = model_config

    def forward(self, model, x0, t, noise_schedule):
        # print(type(noise_schedule))
        assert isinstance(noise_schedule,
                          OrdinalNoiseSchedule), f"{noise_schedule.__class__.__mro__=} \n {type(noise_schedule)=}"
        assert -noise_schedule.num_states / 2 <= x0.min() and x0.max() <= noise_schedule.num_states / 2, f"{x0.min()=} {x0.max()=} {noise_schedule.num_states=}"

        x_t = noise_schedule.sample(t=t, x_0=x0, binomial=False)

        S = noise_schedule.num_states
        norm_fn = lambda x: 2 * x / S
        x_t_norm = norm_fn(x_t)
        x0_norm = norm_fn(x0)
        pred_dict = model(x=x_t_norm, t=t)  # torch.Normal(x_0.loc, x_0.scale)

        if model.output_type == "gaussian":
            '''Model output is N(mu_θ(x), sigma_θ(x))'''
            pred = noise_schedule.unnormalize(pred=pred_dict['gaussian_mu'].clone().detach().cpu())
            pred_dist = torch.distributions.Normal(loc=pred_dict['gaussian_mu'], scale=pred_dict['gaussian_scale'])
            loss = -torch.mean(pred_dist.log_prob(x0_norm))
            assert torch.isnan(loss).sum() == 0.0, "Nan's detected in loss"  # to know which loss is failing
            x_0_t = pred
            output = pred
        elif model.output_type == "logistic":
            '''Model output is probability tensor over x_0 of shape [BS, C, H, W, S_x0]'''
            prob = model.prob_x0_xt(x=x_t_norm, t=t, S=S)['prob']
            pred = torch.argmax(prob, dim=-1) - S / 2
            output = pred
            x_0_t = pred
            x0_long = (x0 + S / 2).long()  # x_0 ∈ {-S/2, ..., S/2} -> x_0 ∈ {0, ..., S}
            # print(f"{pred_dist.shape=} {x0_long.shape=}")
            loss = torch.nn.functional.cross_entropy(input=prob.permute(0, 4, 1, 2, 3), target=x0_long,
                                                     reduction="mean")

        return {"loss": loss, "pred": pred, "target": x0_norm, "x_0": x0, "x_t_norm": x_t, "t": t, 'x_0_t': x_0_t,
                'output': output}
