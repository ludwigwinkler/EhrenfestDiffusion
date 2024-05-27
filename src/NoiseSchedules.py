import einops
import functorch
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.utils
import wandb
from einops import rearrange, repeat
from torch import vmap

from typing import Tuple
import sys


class CategoricalNoiseSchedule:
    def viz_noise_schedule(self, t, data, show=False, title=""):
        p_t = self.p_t(t=t, x0=data)

        x_t = torch.distributions.Categorical(rearrange(p_t, "b c h w -> b h w c")).sample().unsqueeze(1)

        fig, axs = plt.subplots(2, 1)
        axs = axs.flatten()

        t_cont = torch.linspace(start=0.0, end=self.T, steps=500)
        axs[0].plot(t_cont, self.int_beta(t_cont), label="ß")
        axs[0].grid()
        axs[0].set_title(f"{title} ß(t)")
        axs[0].set_ylim(bottom=0.0)

        p_t_fn = lambda t__, x0: self.p_t(t__, x0=x0)
        t_cont = t_cont.unsqueeze(-1)
        x0_ = torch.Tensor([[0.0, 1.0]])
        axs[1].plot(t_cont, vmap(lambda t_: p_t_fn(t_, x0=x0_))(t_cont).squeeze(), label="weight")
        axs[1].set_title("weight w(t)")
        axs[1].set_yticks(torch.linspace(0, 1, 11))
        axs[1].set_ylim(bottom=0.0)
        axs[1].grid()
        plt.tight_layout()
        wandb.log({f"NoiseSchedules/{title} Schedules": wandb.Image(fig)})
        if show:
            plt.show()
        plt.close("all")

        fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
        for p_t_, ax in zip(p_t, axs.flatten()):
            tmp = ax.matshow(p_t_.argmax(dim=0), vmin=0, vmax=data.shape[1])
            ax.set_xticks([]), ax.set_yticks([])
        _ = plt.colorbar(tmp)
        fig.suptitle("Prob of Highest Class")
        plt.tight_layout()
        wandb.log({"NoiseSchedules/Data": wandb.Image(fig)})
        if show:
            plt.show()
        plt.close("all")

        fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
        idx = torch.round(torch.linspace(0, x_t.shape[0] - 1, 25)).int()
        for x_t_, t_, ax in zip(x_t[idx], t[idx], axs.flatten()):
            tmp = ax.matshow(x_t_.squeeze())
            ax.set_title(f"t={t_.item():.2f} ({x_t_.min():.2f}/{x_t_.max():.2f})")
            ax.set_xticks([]), ax.set_yticks([])
        _ = plt.colorbar(tmp)
        plt.suptitle(f"Perturbed Data T={self.T}")
        plt.tight_layout()
        wandb.log({"NoiseSchedules/PerturbedData": wandb.Image(fig)})
        if show:
            plt.show()
        plt.close("all")


class OrdinalNoiseSchedule:
    def viz_noise_schedule(self, t, data, show=False, title=""):
        p_t = self.p_t(t=t, x0=data)
        x_t = torch.distributions.Categorical(p_t).sample()

        fig = plt.figure()
        t, mu_t, std_t = self.compute_moments()
        t, mu_t, std_t = t.squeeze(-1), mu_t.squeeze(-1).T, std_t.squeeze(-1).T
        x0 = torch.arange(0, self.num_states - 1)
        colors = matplotlib.colormaps.get_cmap("jet")(torch.linspace(0, 1, x0.numel()).cpu())
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 1]})
        axs = axs.flatten()
        for color, mu, std in zip(colors, mu_t, std_t):
            axs[0].plot(t.numpy(), mu.numpy())
            axs[0].fill_between(x=t, y1=mu - 3 * std, y2=mu + 3 * std, alpha=0.1)
            probs = torch.distributions.Normal(mu[-1], std[-1] + 1e-8).log_prob(x0).exp()
            axs[1].plot(probs, x0)
        axs[0].set_ylim(0, 256)
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("State #")
        axs[0].set_title("First two moments of Birth Death Process")
        axs[1].set_title(f"Stationary Distribution at {self.T=}")
        plt.suptitle(f"{title}")
        axs[0].grid()
        axs[1].grid()
        plt.show()
        wandb.log({f"NoiseSchedules/{title} Moments": wandb.Image(fig)})
        # plt.close("all")
        # del fig

        if data.shape[1] == 1:
            fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
            idx = torch.round(torch.linspace(0, x_t.shape[0] - 1, 25)).int()
            for x_t_, t_, ax in zip(x_t[idx], t[idx], axs.flatten()):
                tmp = ax.imshow(x_t_.squeeze(), vmin=0.0, vmax=self.num_states)
                ax.set_title(f"t={t_.item():.2f} ({x_t_.min():.2f}/{x_t_.max():.2f})")
                ax.set_xticks([]), ax.set_yticks([])
            _ = plt.colorbar(tmp)

        elif data.shape[1] == 3:
            fig = plt.figure()
            grid = einops.rearrange(torchvision.utils.make_grid(x_t[:: (x_t.shape[0] // 64)]), "c ... -> ... c").numpy()
            plt.imshow(grid)
            plt.suptitle(f"Data:{data.shape}; Every {x_t.shape[0] // 64}th image shown")

        plt.suptitle(f"{title} Perturbed Data T={self.T}")
        plt.tight_layout()
        wandb.log({f"NoiseSchedules/{title}PerturbedData": wandb.Image(fig)})
        plt.show()
        # plt.close("all")
        # del fig

        if hasattr(self, "int_beta_t"):
            t = torch.linspace(0, self.T, 100)
            int_beta_t = self.int_beta_t(t)
            fig = plt.figure()
            plt.plot(t, int_beta_t)
            plt.title(f"{title} Rate (min:{int_beta_t.min().item():.2f})")
            wandb.log({f"NoiseSchedules/{title} Rate": wandb.Image(fig)})
            plt.show()
        # plt.close("all")
        # del fig

        if show:
            plt.show()


class BinaryManfred(torch.nn.Module, CategoricalNoiseSchedule):
    def __init__(self, T):
        super(torch.nn.Module, CategoricalNoiseSchedule).__init__()
        self.T = T

    def int_beta(self, t):
        return torch.clip((1 - (torch.cos(t / self.T * torch.pi / 2) + 1e-5) ** 0.5), min=1e-10)

    def weight(self, t, x0=None):
        weight = torch.exp(-2 * (1 - (torch.cos(t / self.T * torch.pi / 2) + 1e-5) ** 0.5))
        if x0 is None:
            return weight
        else:
            return torch.einsum("b ..., b -> b ...", x0, weight)


class TimeEhrenfest(OrdinalNoiseSchedule):
    '''
    https://en.wikipedia.org/wiki/Ehrenfest_model
    x ∈ [0, S] -> x ∈ [-S/2 + ϵ, S/2 + ϵ]
    r^+(x) = λ_t ( S_expanded_half + x) = λ_t ( (S/2 + ϵ) + x)
    r^-(x) = λ_t ( S_expanded_half - x) = λ_t ( (S/2 + ϵ) - x)
    We want e^{-2 λ_t} = cos(pi/2 * t/T)^2

    Moments:
        E[x_t_norm | x_0] = x_0 * e^(-2 λ_t) = x_0 * w_t
        V[x_t_norm | x_0] = (1-e^(-4 λ_t)) * S_expanded_half/2
    '''

    def __init__(self, num_states, cfg):
        super().__init__()

        self.cfg = cfg
        self.T = cfg.T
        self.num_states = num_states
        self.S = num_states
        self.S_expanded_half = (num_states / 2 + cfg.epsilon_T)
        self.hparams = cfg
        self.std_T = (self.S_expanded_half / 2) ** 0.5

    def __repr__(self):
        return (
            f"Time Dependent Ehrenfest Process \n "
            f"\t num_states={self.S} \n"
            # f"\t lambda_0={self.epsilon_0} \n "
            f"\t Expanded States (half) = (S + ϵ)/2 ={self.S_expanded_half} \n"
            f"\t Sdt_T={self.std_T:.2f} \n"
            f"\t T={self.T} \n"
        )

    def w_t(self, t):
        '''
        e^(-2 λ_t) = cos(π/2 * t/T)^2
        e^(-2 λ_t) = 1- t/T
        '''
        w_t = torch.cos(torch.pi / 2 * t).pow(2)  # + 0.001 * t
        # w_t = (1 - (t / self.T)).clamp(1e-5, 0.999)
        return w_t

    @torch.enable_grad()
    def lambda_t(self, t):
        '''
        μ = x_0 * e^(-2 λ_t) = x_0 * w_t
        w_t = e^(-2 λ_t)
        λ_t = - 1/2 log(w_t)
        '''
        # dlambda_t = functorch.grad(lambda t: - 0.5 * torch.log(self.w_t(t)))(t)
        # t = t.double()
        t = t.clamp(0.01, 0.99)

        if not t.requires_grad:
            t.requires_grad_(True)
        lambda_t = - 0.5 * torch.log(self.w_t(t))
        dlambda_t = torch.autograd.grad(lambda_t, inputs=t, grad_outputs=torch.ones_like(t))[0]
        # dlambda_t = torch.pi / 2 * torch.tan(torch.pi / 2 * t)
        # dlambda_t = torch.pi / 2 * torch.sin(torch.pi / 2 * t) / (torch.sin(torch.pi / 2 * t) + 0.001)
        return dlambda_t.float()  # .clamp(0.0001, 500)

    def increment_forward_rate(self, t, x_t):
        # return 10 * (self.S / 2 - x_t_norm).clamp(min=0.) + 1e-5  # + self.epsilon_t(t)
        return self.lambda_t(t) * (self.S_expanded_half - x_t).clamp(min=0.)

    def decrement_forward_rate(self, t, x_t):
        # return 10 * (x_t_norm + self.S / 2).clamp(min=0.) + 1e-5  # + self.epsilon_t(t)
        return self.lambda_t(t) * (self.S_expanded_half + x_t).clamp(min=0.)

    def reverse_rates(self, t, x_t, pred) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        r^+(x) = r^-(x+1) * E_pred [ q(x+1|x_0) / q(x|x_0) ]
        r^-(x) = r^+(x-1) * E_pred [ q(x-11|x_0) / q(x|x_0) ]
        '''
        S = self.S
        # assert -S / 2 <= x_t_norm.min() and x_t_norm.max() <= S / 2, f"Ehrenfest CTMC sample_next_state: {x_t_norm.min()=} {x_t_norm.max()=}"
        t.requires_grad_()
        assert isinstance(pred, dict)

        s = torch.arange(-S / 2, S / 2, device=x_t.device, dtype=x_t.dtype)  # S=256 -> [-127, 128]
        # print(f"{s=}")
        # sys.exit()
        x0_s = torch.ones_like(x_t).unsqueeze(-1) * s  # [-S/2, ..., S/2] of shape [BS, C, H, W, S]
        '''
        Compute moments of all x_0: μ(x_0, t), σ(x_0, t)
        '''
        _, mu_x0_s, std_x0_s = self.compute_moments(t=t, x0=x0_s)

        if 'prob' in pred:
            assert 'gaussian_mu' in pred or 'logistic_mu' in pred
            '''p(x_0 | x_t_norm) under variational distribution'''
            prob = pred['prob']
            '''Σ_x0 p(x_0 | x_t_norm) * r^+(x_t_norm)'''
            birth_ratio = torch.exp(
                (-2 * (x_t.unsqueeze(-1) - mu_x0_s) - 1) / (2 * std_x0_s.pow(2)))  # [BS, C, H, W, S]
            birth_rate = torch.sum(prob * birth_ratio, dim=-1)
            '''Σ_x0 p(x_0 | x_t_norm) * r^-(x_t_norm)'''
            death_ratio = torch.exp((2 * (x_t.unsqueeze(-1) - mu_x0_s) - 1) / (2 * std_x0_s.pow(2)))  # [BS, C, H, W, S]
            death_rate = torch.sum(prob * death_ratio, dim=-1)


        elif 'taylor1' in pred:
            '''
            Manfred's Taylor expansion for deterministic rate prediction
            r_backward^±(x) = r_forward^∓(x ± 1) exp[ -1/σ^2_t ] ( 1 ∓ (x - exp[ -2λ_t ] E_p(x_0|x_t)[x_0]) / σ_t^2 )    
            '''
            _, _, std_x0 = self.compute_moments(t=t, x0=pred['taylor1'])
            var_x0 = std_x0.pow(2)
            taylor1 = self.unnormalize(pred['taylor1'])
            birth_rate = 1 - (x_t - self.w_t(t) * taylor1) / var_x0
            death_rate = 1 + (x_t - self.w_t(t) * taylor1) / var_x0
            birth_rate = torch.exp(-1 / var_x0) * birth_rate
            death_rate = torch.exp(-1 / var_x0) * death_rate
        elif 'ratio' in pred:
            # 1 UNet
            output = pred['ratio']
            death_rate, birth_rate = output.chunk(2, dim=1)
        elif 'ratio2' in pred:
            # 2 UNets
            output = pred['ratio2']
            death_rate, birth_rate = output.chunk(2, dim=1)
        elif 'x0' in pred:
            x0_ = self.unnormalize(pred['x0'])
            _, _, std_x0 = self.compute_moments(t=t, x0=x0_)
            var_x0 = std_x0.pow(2)
            birth_rate = torch.exp(-2 * (x_t - self.w_t(t) * x0_) / (2 * var_x0)).clamp(min=0., max=200)
            death_rate = torch.exp(2 * (x_t - self.w_t(t) * x0_) / (2 * var_x0)).clamp(min=0., max=200)
        elif 'score' in pred:
            '''
            Network trained on MSE(score - (- (x-mu)/std)
            Prediction has to be multiplied by additional 1/ std
            '''

            _, _, std = self.compute_moments(t=t, x0=x_t)  # check huggingface predictor, it's actually x0
            score = pred['score'] / std
            birth_rate = 1 + score
            death_rate = 1 - score
        elif 'score_x0' in pred:
            x0 = self.unnormalize(pred['score_x0'])
            _, mu, std = self.compute_moments(t=t, x0=x0)  # check huggingface predictor, it's actually x0
            score = -(x_t - mu) / std ** 2
            birth_rate = 1 + score
            death_rate = 1 - score

        """Multiply by corresponding forward rate."""
        birth_rate = self.decrement_forward_rate(t, x_t + 1) * birth_rate
        death_rate = self.increment_forward_rate(t, x_t - 1) * death_rate

        death_rate = torch.nan_to_num(death_rate, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=1e-3)
        birth_rate = torch.nan_to_num(birth_rate, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=1e-3)

        return birth_rate, death_rate

    def forward_rates(self, t, x_t):
        return self.increment_forward_rate(t, x_t), self.decrement_forward_rate(t, x_t)

    @torch.no_grad()
    def sample(self, t, x_0, binomial=True):
        '''
        μ(t, x_0) = e^{-2 \lambda(t)} * x_0 = w(t) * x_0
        '''
        if binomial:
            '''
            self.S_expanded_half = (S + epsilon)/2
            Total space is 2 * S_expanded_half = S + epsilon
            '''
            S = self.S
            B, C, H, W = x_0.shape
            f_t = 0.5 * (1 + self.w_t(t))
            x_0_e = x_0 + self.S_expanded_half  # [-S_expanded_half, ..., S_expanded_half] -> [0, ..., S+epsilon]
            f_t = einops.repeat(f_t, "B -> B C H W", C=C, H=H, W=W)
            bin0 = torch.binomial(2 * self.S_expanded_half - x_0_e, 1 - f_t)
            bin1 = torch.binomial(x_0_e, f_t)
            sample = bin0 + bin1 - self.S_expanded_half
            # print(f"Binomial {sample.shape=} {sample.min()=} {sample.max()=}")
        else:  # gaussian
            t, mu, std = self.compute_moments(t=t.float(), x0=x_0.float())
            sample = mu + std * torch.randn_like(mu)
            # print(f"Gaussian {sample.shape=} {sample.min()=} {sample.max()=}")
        return sample

    def compute_moments(self, t=None, x0=None):

        if t is None and x0 is None:
            x0 = torch.arange(0, self.S)[:: self.S // 5].reshape(-1, 1) - self.S / 2 + 1
            t = torch.linspace(0, self.T, 1000).reshape(-1, 1)
            mu = torch.vmap(lambda t_: torch.vmap(lambda x0_: x0_ * self.w_t(t_))(x0))(t)
            var = (1 - self.w_t(t).pow(2)) * self.S_expanded_half / 2
            std = var.pow(0.5).unsqueeze(-1) * torch.ones_like(mu)
        else:
            assert t.shape == x0.shape[:t.ndim] or t.shape == x0.shape, f"{t.shape=} {x0.shape=}"
            if t.shape == torch.Size((x0.shape[0],)):  # [BS] , [BS, C, H, W]
                contraction_str = 'b ..., b -> b ...'
            elif t.shape == x0.shape:  # [BS, C, H, W] , [BS, C, H, W]
                contraction_str = 'b ..., b ... -> b ...'
            elif t.shape == x0.shape[:t.ndim] and t.ndim < x0.ndim:  # [BS, C, H, W] , [BS, C, H, W, S]
                contraction_str = 'b c h w ..., b c h w -> b c h w ...'
            mu = einops.einsum(x0, self.w_t(t), contraction_str)
            var = (1 - self.w_t(t).pow(2)) * self.S_expanded_half / 2 + 0.01
            std = einops.einsum(torch.ones_like(mu), var.pow(0.5), contraction_str)
            assert x0.shape == mu.shape == std.shape, f"Ehrenfest.compute_moments(): {x0.shape=} {mu.shape=} {std.shape=}"
            std = torch.nan_to_num(std, nan=1., posinf=1., neginf=1.).clamp(min=1e-3)
            mu = torch.nan_to_num(mu, nan=0, posinf=0, neginf=0.).clamp(min=1e-3)
            assert torch.isnan(std).sum() == 0, f"NaN in std detected {std}"
            assert torch.isnan(mu).sum() == 0, f"NaN in std detected {mu}"
        return t, mu, std + 0.01

    def normalize(self, pred):
        if isinstance(pred, torch.Tensor):
            return pred * 2 / self.S
        if isinstance(pred, torch.distributions.Normal):
            loc = pred.loc * 2 / self.S
            scale = pred.scale * 6 / self.S
            return torch.distributions.Normal(loc=loc, scale=scale, validate_args=False)
        else:
            raise NotImplementedError(f"unnormalize() not implemented for {type(pred)}")

    def unnormalize(self, pred):
        if isinstance(pred, torch.Tensor):
            return pred * self.S / 2
        else:
            raise NotImplementedError(f"unnormalize() not implemented for {type(pred)}")

    def viz_noise_schedule(self, t, data, show=False, title=""):
        # p_t = self.p_t(t=t, x_0=data)
        # x_t_norm = torch.distributions.Categorical(p_t).sample() - self.S / 2 + 1

        print(self)
        scores = []
        ratios = []
        for t_ in [0.0, 0.01, 0.25, 0.5, 0.75, 1.0]:
            print(f"t={t_}")
            equilibrium_samples_binomial = self.sample(t=torch.ones_like(t) * t_, x_0=data, binomial=True)
            print(
                f"Empirical Equilibrium Distribution (Binom) t={t_}: μ:{equilibrium_samples_binomial.mean().item():.3f} σ:{equilibrium_samples_binomial.std().item():.3f} Min/Max:{equilibrium_samples_binomial.max().item():.3f} {equilibrium_samples_binomial.max().item()}")
            equilibrium_samples_normal = self.sample(t=torch.ones_like(t) * t_, x_0=data, binomial=False)
            print(
                f"Empirical Equilibrium Distribution (Normal) t={t_}: μ:{equilibrium_samples_normal.mean().item():.3f} σ:{equilibrium_samples_normal.std().item():.3f} Min/Max:{equilibrium_samples_normal.int().max().item():.3f} {equilibrium_samples_normal.int().max().item()}")
            print()
            _, mu, std = self.compute_moments(t=torch.ones_like(t) * t_, x0=data)
            scores += [{'t': t_, 'score': -(equilibrium_samples_normal - mu) / std ** 2}]
            ratios += [{'t': t_,
                        'death_rate': torch.exp((1 - 2 * (equilibrium_samples_binomial - mu)) * (-1 / std ** 2)),
                        'birth_rate': torch.exp((1 + 2 * (equilibrium_samples_binomial - mu)) * (-1 / std ** 2))}]

        sampled_moments = {'gaussian': {'mu': [], 'std': []}, 'binomial': {'mu': [], 'std': []}, 't': []}
        for t_ in torch.linspace(0, 1, 25):
            equilibrium_samples_binomial = self.sample(t=torch.ones_like(t) * t_, x_0=data, binomial=True)
            equilibrium_samples_normal = self.sample(t=torch.ones_like(t) * t_, x_0=data, binomial=False)

            sampled_moments['gaussian']['mu'].append(equilibrium_samples_normal.mean().item())
            sampled_moments['gaussian']['std'].append(equilibrium_samples_normal.std().item())
            sampled_moments['binomial']['mu'].append(equilibrium_samples_binomial.mean().item())
            sampled_moments['binomial']['std'].append(equilibrium_samples_binomial.std().item())
            sampled_moments['t'].append(t_.item())

        min_val = -self.S / 2 + 1
        max_val = self.S / 2

        fig = plt.figure()
        t_, mu_t, std_t = self.compute_moments()
        t_, mu_t, std_t = t_.squeeze(-1), mu_t.squeeze(-1).T, std_t.squeeze(-1).T
        x0 = torch.arange(-self.S / 2 + 1, self.S / 2)
        colors = matplotlib.colormaps.get_cmap("jet")(torch.linspace(0, 1, x0.numel()).cpu())
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={"width_ratios": [3, 1]})
        axs = axs.flatten()
        for color, mu, std in zip(colors, mu_t, std_t):
            axs[0].plot(t_.numpy(), mu.numpy())
            axs[0].fill_between(x=t_, y1=mu - 3 * std, y2=mu + 3 * std, alpha=0.1)
            probs = torch.distributions.Normal(mu[-1], std[-1] + 1e-8).log_prob(x0).exp()
            axs[1].plot(probs, x0)
        axs[2].plot(t_.detach().cpu(), self.lambda_t(t_).detach().cpu())
        axs[3].plot(t_.detach().cpu(), self.lambda_t(t_).detach().cpu())
        axs[0].set_ylim(-0.75 * self.S, 0.75 * self.S)
        axs[1].set_ylim(-0.75 * self.S, 0.75 * self.S)
        axs[2].set_ylim(-0.1, 10)
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("State #")
        axs[0].set_title("First two moments of Birth Death Process")
        axs[1].set_title(f"Stationary Distribution at {self.T=}")
        plt.suptitle(f"{title}")
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()
        [ax.grid() for ax in axs.flatten()]
        plt.show()
        wandb.log({f"NoiseSchedule/{title} Moments": wandb.Image(fig)})

        t, mu_, std_ = self.compute_moments(t=t, x0=data)
        x_t_normal = self.sample(t=t, x_0=data, binomial=False)
        x_t_binomial = self.sample(t=t, x_0=data, binomial=True)

        for x_t, name in zip([x_t_normal, x_t_binomial], ["Normal", "Binomial"]):
            if data.shape[1] == 1:
                fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
                idx = torch.round(torch.linspace(0, x_t.shape[0] - 1, 25)).int()
                for x_t_, t_, ax in zip(x_t[idx], t[idx], axs.flatten()):
                    tmp = ax.imshow(x_t_.squeeze(), vmin=min_val, vmax=max_val)
                    ax.set_title(f"t={t_.item():.2f} ({x_t_.min():.2f}/{x_t_.max():.2f})")
                    ax.set_xticks([]), ax.set_yticks([])
                _ = plt.colorbar(tmp)

            elif data.shape[1] == 3:
                fig = plt.figure()
                x_t_ = (x_t[:: x_t.shape[0] // min(64, x_t.shape[0])] + self.S / 2) / self.S
                grid = einops.rearrange(torchvision.utils.make_grid(x_t_), "c ... -> ... c").cpu().numpy()
                plt.imshow(grid)
                plt.suptitle(f"Data:{data.shape}; Every {x_t.shape[0] // 64}th image shown")

            plt.suptitle(f"{title} Perturbed Data T={self.T} ({name} sampling)")
            plt.tight_layout()
            plt.grid(False)
            wandb.log({f"NoiseSchedule/{title}PerturbedData ({name} sampling)": wandb.Image(fig)})
            plt.show()
            # plt.close("all")
            # del fig

        if hasattr(self, "int_beta_t"):
            t = torch.linspace(0, self.T, 100)
            int_beta_t = self.int_beta_t(t)
            fig = plt.figure()
            plt.plot(t, int_beta_t)
            plt.title(f"{title} Rate (min:{int_beta_t.min().item():.2f})")
            wandb.log({f"NoiseSchedules/{title} Rate": wandb.Image(fig)})
            plt.show()
        # plt.close("all")
        # del fig

        if show:
            plt.show()
        plt.close("all")


class ScaledEhrenfest(OrdinalNoiseSchedule):
    '''
    https://en.wikipedia.org/wiki/Ehrenfest_model
    x ∈ [0, S] -> x ∈ [-S/2 + ϵ, S/2 + ϵ]
    r^+(x) = λ_t ( S_expanded_half + x) = λ_t ( (S/2 + ϵ) + x)
    r^-(x) = λ_t ( S_expanded_half - x) = λ_t ( (S/2 + ϵ) - x)
    We want e^{-2 λ_t} = cos(pi/2 * t/T)^2

    Moments:
        E[x_t_norm | x_0] = x_0 * e^(-2 λ_t) = x_0 * w_t
        V[x_t_norm | x_0] = (1-e^(-4 λ_t)) * S_expanded_half/2
    '''

    def __init__(self, num_states, cfg):
        super().__init__()

        self.cfg = cfg
        self.T = cfg.T
        self.num_states = num_states
        self.S = num_states
        self.hparams = cfg

    def __repr__(self):
        return (
            f"Scaled Time Dependent Ehrenfest Process \n "
            f"\t num_states={self.S} \n"
            f"\t SDE: {self.hparams.schedule_type}"
        )

    def w_t(self, t):
        '''
        e^(-2 ∫ λ_s ds) = cos(π/2 * t/T)^2
        e^(-2 ∫ λ_s ds) = e^(- 1/4 t^2 (ß_max - ß_min) - 1/2 t ß_min)
        '''
        if self.hparams.schedule_type == "cosine":
            w_t = torch.cos(torch.pi / 2 * t).pow(2)  # + 0.001 * t
        elif self.hparams.schedule_type == "song":
            beta_max, beta_min = self.hparams.song_beta_max, self.hparams.song_beta_min
            w_t = torch.exp(-1 / 4 * t ** 2 * (beta_max - beta_min) - 1 / 2 * t * beta_min)
        return w_t

    @torch.enable_grad()
    def lambda_t(self, t):
        '''
        μ = x_0 * e^(-2 λ_t) = x_0 * w_t
        w_t = e^(-2 λ_t)
        λ_t = - 1/2 log(w_t)
        '''
        # dlambda_t = functorch.grad(lambda t: - 0.5 * torch.log(self.w_t(t)))(t)
        # t = t.double()
        t = t.clamp(0.01, 0.99)

        if self.hparams.schedule_type == "cosine":

            dlambda_t = torch.pi * torch.tan(torch.pi / 2 * t)
        elif self.hparams.schedule_type == "song":
            beta_max, beta_min = self.hparams.song_beta_max, self.hparams.song_beta_min
            dlambda_t = beta_min + t * (beta_max - beta_min)
            dlambda_t = 1 / 2 * dlambda_t

        return dlambda_t.float().clamp(0.0001, 1000)

    def increment_forward_rate(self, t, x_t):
        # raise NotImplementedError('ScaledEhrenfest.increment_forward_rate()')

        return self.lambda_t(t) * self.S / 4 * (self.S - x_t)

    def decrement_forward_rate(self, t, x_t):
        # raise NotImplementedError('ScaledEhrenfest.decrement_forward_rate()')

        return self.lambda_t(t) * self.S / 4 * (self.S + x_t)

    def reverse_rates(self, t, x_t, pred) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        r^+(x) = r^-(x+1) * E_pred [ q(x+1|x_0) / q(x|x_0) ]
        r^-(x) = r^+(x-1) * E_pred [ q(x-11|x_0) / q(x|x_0) ]
        '''
        S = self.S
        scaling = 2 / self.S
        # assert -S / 2 <= x_t_norm.min() and x_t_norm.max() <= S / 2, f"Ehrenfest CTMC sample_next_state: {x_t_norm.min()=} {x_t_norm.max()=}"
        t.requires_grad_()
        assert isinstance(pred, dict)

        s = torch.linspace(-1, 1, steps=S, device=x_t.device, dtype=x_t.dtype)  # S=256 -> [-127, 128]
        # print(f"{s=}")
        # sys.exit()
        x0_s = torch.ones_like(x_t).unsqueeze(-1) * s  # [-S/2, ..., S/2] of shape [BS, C, H, W, S]
        '''
        Compute moments of all x_0: μ(x_0, t), σ(x_0, t)
        '''
        _, mu_x0_s, std_x0_s = self.compute_moments(t=t, x0=x0_s)
        w_t = self.w_t(t)
        if w_t.ndim == 1:
            w_t = w_t.view(-1, 1, 1, 1)  # [BS] -> [BS, C=1, H=1, W=1]

        if 'prob' in pred:
            assert 'gaussian_mu' in pred or 'logistic_mu' in pred
            '''p(x_0 | x_t_norm) under variational distribution'''
            prob = pred['prob']
            '''Σ_x0 p(x_0 | x_t_norm) * r^+(x_t_norm)'''
            birth_ratio = torch.exp(
                (-2 * (x_t.unsqueeze(-1) - mu_x0_s) - 1) / (2 * std_x0_s.pow(2)))  # [BS, C, H, W, S]
            birth_rate = torch.sum(prob * birth_ratio, dim=-1)
            '''Σ_x0 p(x_0 | x_t_norm) * r^-(x_t_norm)'''
            death_ratio = torch.exp((2 * (x_t.unsqueeze(-1) - mu_x0_s) - 1) / (2 * std_x0_s.pow(2)))  # [BS, C, H, W, S]
            death_rate = torch.sum(prob * death_ratio, dim=-1)

        elif 'epsilon' in pred:
            if self.hparams.rate == "score":
                _, _, std = self.compute_moments(t=t, x0=pred['epsilon'])
                score = - pred['epsilon'] / std
                x0 = (x_t - std * pred['epsilon'].detach()) / w_t
                mu = w_t * x0
                birth_rate = 1 + scaling * score
                death_rate = 1 - scaling * score
            elif self.hparams.rate == 'taylor1':
                '''We learned epsilon'''
                _, _, std = self.compute_moments(t=t, x0=pred['epsilon'])
                var = std.pow(2)
                x0 = (x_t - std * pred['epsilon'].detach()) / w_t
                mu = w_t * x0
                birth_rate = 1 - scaling * (x_t - mu) / var
                death_rate = 1 + scaling * (x_t - mu) / var
                birth_rate = torch.exp(-scaling ** 2 / (2 * var)) * birth_rate
                death_rate = torch.exp(-scaling ** 2 / (2 * var)) * death_rate
            elif self.hparams.rate == 'taylor2':
                '''We learned epsilon'''
                _, _, std = self.compute_moments(t=t, x0=pred['epsilon'])
                var = std.pow(2)
                x0 = (x_t - std * pred['epsilon'].detach()) / w_t
                mu = w_t * x0
                birth_rate = 1 - scaling * (x_t - mu) / var
                death_rate = 1 + scaling * (x_t - mu) / var
                second_order = scaling ** 2 * (x_t - mu) ** 2 / (2 * var ** 2)
                birth_rate += second_order
                death_rate += second_order
                birth_rate = torch.exp(-scaling ** 2 / (2 * var)) * birth_rate
                death_rate = torch.exp(-scaling ** 2 / (2 * var)) * death_rate
            else:
                raise NotImplementedError(f'Cant predict {self.hparams.rate} from epsilon prediction')
        elif 'score' in pred:
            if self.hparams.rate == 'score':
                _, _, std = self.compute_moments(t=t, x0=pred['score'])
                score = pred['score']
                birth_rate = 1 + scaling * score
                death_rate = 1 - scaling * score
            elif self.hparams.rate == 'taylor1':
                '''Predicting mu(x_0)'''
                _, _, std = self.compute_moments(t=t, x0=pred['score'])
                var = std.pow(2)
                score = pred['score']
                birth_rate = 1 + scaling * score
                death_rate = 1 - scaling * score
                birth_rate = torch.exp(-scaling ** 2 / (2 * var)) * birth_rate
                death_rate = torch.exp(-scaling ** 2 / (2 * var)) * death_rate
            else:
                raise NotImplementedError(f'Cant predict {self.hparams.rate} from epsilon prediction')

        elif 'taylor1' in pred:
            if self.hparams.rate == 'score':
                _, _, std = self.compute_moments(t=t, x0=pred['taylor1'])
                score = - (x_t - pred['taylor1']) / std.pow(2)
                x0 = pred['taylor1'].detach() / w_t
                birth_rate = 1 + scaling * score
                death_rate = 1 - scaling * score
            elif self.hparams.rate == 'taylor1':
                '''Predicting mu(x_0)'''
                _, _, std = self.compute_moments(t=t, x0=pred['taylor1'])
                var = std.pow(2)
                score = -(x_t - pred['taylor1']) / var
                birth_rate = 1 + scaling * score
                death_rate = 1 - scaling * score
                birth_rate = torch.exp(-scaling ** 2 / (2 * var)) * birth_rate
                death_rate = torch.exp(-scaling ** 2 / (2 * var)) * death_rate
            else:
                raise NotImplementedError(f'Cant predict {self.hparams.rate} from epsilon prediction')
        elif pred['output_type'] == 'taylor2':
            _, _, std = self.compute_moments(t=t, x0=pred['output'])

            var = std.pow(2)
            # print(f"{x_t.shape=} {std.shape=} {pred['epsilon'].shape=} {w_t.shape=}")
            x0 = (x_t - std * pred['epsilon'].detach()) / w_t
            mu = w_t * x0

            birth_rate = 1 - scaling * (x_t - mu) / var
            death_rate = 1 + scaling * (x_t - mu) / var
            second_order = scaling ** 2 * (x_t - mu) ** 2 / (2 * var ** 2)
            birth_rate += second_order
            death_rate += second_order
            birth_rate = torch.exp(-scaling ** 2 / (2 * var)) * birth_rate
            death_rate = torch.exp(-scaling ** 2 / (2 * var)) * death_rate
        elif 'ratio' in pred:
            # 1 UNet
            output = pred['ratio']
            death_rate, birth_rate = output.chunk(2, dim=1)
        elif 'ratio2' in pred:
            # 2 UNets
            output = pred['ratio2']
            death_rate, birth_rate = output.chunk(2, dim=1)
        elif 'x0' in pred:
            x0 = pred['x0']
            _, _, std = self.compute_moments(t=t, x0=x0)
            var_x0 = std.pow(2)
            birth_rate = torch.exp(-2 * (x_t - w_t * x0) / (2 * var_x0)).clamp(min=0., max=200)
            death_rate = torch.exp(2 * (x_t - w_t * x0) / (2 * var_x0)).clamp(min=0., max=200)

        """Multiply by corresponding forward rate."""
        # print(f"{death_rate.shape=} {birth_rate.shape=}")
        birth_rate = self.decrement_forward_rate(t, x_t) * birth_rate
        death_rate = self.increment_forward_rate(t, x_t) * death_rate

        death_rate = torch.nan_to_num(death_rate, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=1e-3)
        birth_rate = torch.nan_to_num(birth_rate, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=1e-3)

        return death_rate, birth_rate

    def forward_rates(self, t, x_t):
        return self.increment_forward_rate(t, x_t), self.decrement_forward_rate(t, x_t)

    @torch.no_grad()
    def sample(self, t, x_0, binomial=False):
        '''
        μ(t, x_0) = e^{-2 \lambda(t)} * x_0 = w(t) * x_0
        '''
        if binomial:
            '''
            self.S_expanded_half = (S + epsilon)/2
            Total space is 2 * S_expanded_half = S + epsilon
            '''
            S = self.S
            B, C, H, W = x_0.shape
            f_t = 0.5 * (1 + self.w_t(t))
            x_0_e = x_0 + self.S_expanded_half  # [-S_expanded_half, ..., S_expanded_half] -> [0, ..., S+epsilon]
            f_t = einops.repeat(f_t, "B -> B C H W", C=C, H=H, W=W)
            bin0 = torch.binomial(2 * self.S_expanded_half - x_0_e, 1 - f_t)
            bin1 = torch.binomial(x_0_e, f_t)
            sample = bin0 + bin1 - self.S_expanded_half
            # print(f"Binomial {sample.shape=} {sample.min()=} {sample.max()=}")
        else:  # gaussian
            t, mu, std = self.compute_moments(t=t.float(), x0=x_0.float())
            sample = mu + std * torch.randn_like(mu)
            # print(f"Gaussian {sample.shape=} {sample.min()=} {sample.max()=}")
        return sample

    def compute_moments(self, t=None, x0=None):

        if t is None and x0 is None:
            x0 = torch.linspace(-1, 1, steps=self.S)[:: self.S // 5].reshape(-1, 1)
            t = torch.linspace(0, self.T, 1000).reshape(-1, 1)
            mu = torch.vmap(lambda t_: torch.vmap(lambda x0_: x0_ * self.w_t(t_))(x0))(t)
            var = (1 - self.w_t(t).pow(2))
            std = var.pow(0.5).unsqueeze(-1) * torch.ones_like(mu)
        else:
            assert t.shape == x0.shape[:t.ndim] or t.shape == x0.shape, f"{t.shape=} {x0.shape=}"
            if t.shape == torch.Size((x0.shape[0],)):  # [BS] , [BS, C, H, W]
                contraction_str = 'b ..., b -> b ...'
            elif t.shape == x0.shape:  # [BS, C, H, W] , [BS, C, H, W]
                contraction_str = 'b ..., b ... -> b ...'
            elif t.shape == x0.shape[:t.ndim] and t.ndim < x0.ndim:  # [BS, C, H, W] , [BS, C, H, W, S]
                contraction_str = 'b c h w ..., b c h w -> b c h w ...'
            mu = einops.einsum(x0, self.w_t(t), contraction_str)
            var = (1 - self.w_t(t).pow(2)) + 0.00001
            std = einops.einsum(torch.ones_like(mu), var.pow(0.5), contraction_str)
            assert x0.shape == mu.shape == std.shape, f"Ehrenfest.compute_moments(): {x0.shape=} {mu.shape=} {std.shape=}"
            # std = torch.nan_to_num(std, nan=1., posinf=1., neginf=1.).clamp(min=1e-3)
            # mu = torch.nan_to_num(mu, nan=0, posinf=0, neginf=0.).clamp(min=1e-3)
            assert torch.isnan(std).sum() == 0, f"NaN in std detected {std}"
            assert torch.isnan(mu).sum() == 0, f"NaN in std detected {mu}"
        return t, mu, std

    def viz_noise_schedule(self, t, data, show=False, title=""):

        print(self)
        scores = []
        ratios = []
        for t_ in [0.0, 0.01, 0.25, 0.5, 0.75, 1.0]:
            print(f"t={t_}")
            equilibrium_samples_binomial = self.sample(t=torch.ones_like(t) * t_, x_0=data, binomial=False)
            print(
                f"Empirical Equilibrium Distribution (Binom) t={t_}: μ:{equilibrium_samples_binomial.mean().item():.3f} σ:{equilibrium_samples_binomial.std().item():.3f} Min/Max:{equilibrium_samples_binomial.max().item():.3f} {equilibrium_samples_binomial.max().item()}")
            equilibrium_samples_normal = self.sample(t=torch.ones_like(t) * t_, x_0=data, binomial=False)
            print(
                f"Empirical Equilibrium Distribution (Normal) t={t_}: μ:{equilibrium_samples_normal.mean().item():.3f} σ:{equilibrium_samples_normal.std().item():.3f} Min/Max:{equilibrium_samples_normal.int().max().item():.3f} {equilibrium_samples_normal.int().max().item()}")
            print()
            _, mu, std = self.compute_moments(t=torch.ones_like(t) * t_, x0=data)
            scores += [{'t': t_, 'score': -(equilibrium_samples_normal - mu) / std ** 2}]
            ratios += [{'t': t_,
                        'death_rate': torch.exp((1 - 2 * (equilibrium_samples_binomial - mu)) * (-1 / std ** 2)),
                        'birth_rate': torch.exp((1 + 2 * (equilibrium_samples_binomial - mu)) * (-1 / std ** 2))}]

        sampled_moments = {'gaussian': {'mu': [], 'std': []}, 'binomial': {'mu': [], 'std': []}, 't': []}
        for t_ in torch.linspace(0, 1, 25):
            equilibrium_samples_binomial = self.sample(t=torch.ones_like(t) * t_, x_0=data, binomial=False)
            equilibrium_samples_normal = self.sample(t=torch.ones_like(t) * t_, x_0=data, binomial=False)

            sampled_moments['gaussian']['mu'].append(equilibrium_samples_normal.mean().item())
            sampled_moments['gaussian']['std'].append(equilibrium_samples_normal.std().item())
            sampled_moments['binomial']['mu'].append(equilibrium_samples_binomial.mean().item())
            sampled_moments['binomial']['std'].append(equilibrium_samples_binomial.std().item())
            sampled_moments['t'].append(t_.item())

        # min_val = -self.S / 2 + 1
        # max_val = self.S / 2

        fig = plt.figure()
        t_, mu_t, std_t = self.compute_moments()
        t_, mu_t, std_t = t_.squeeze(-1), mu_t.squeeze(-1).T, std_t.squeeze(-1).T
        x0 = torch.linspace(-3, 3, steps=self.S)
        colors = matplotlib.colormaps.get_cmap("jet")(torch.linspace(0, 1, x0.numel()).cpu())
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={"width_ratios": [3, 1]})
        axs = axs.flatten()
        for color, mu, std in zip(colors, mu_t, std_t):
            axs[0].plot(t_.numpy(), mu.numpy())
            axs[0].fill_between(x=t_, y1=mu - 3 * std, y2=mu + 3 * std, alpha=0.1)
            probs = torch.distributions.Normal(mu[-1], std[-1] + 1e-8).log_prob(x0).exp()
            axs[1].plot(probs, x0)
        axs[2].plot(t_.detach().cpu(), self.lambda_t(t_).detach().cpu())
        axs[3].plot(t_.detach().cpu(), self.lambda_t(t_).detach().cpu())
        axs[0].set_ylim(-5, 5)
        axs[1].set_ylim(-5, 5)
        axs[2].set_ylim(-0.1, 10)
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("State #")
        axs[0].set_title("First two moments of Birth Death Process")
        axs[1].set_title(f"Stationary Distribution at {self.T=}")
        plt.suptitle(f"{title}")
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()
        [ax.grid() for ax in axs.flatten()]
        plt.show()
        wandb.log({f"NoiseSchedule/{title} Moments": wandb.Image(fig)})

        t, mu_, std_ = self.compute_moments(t=t, x0=data)
        x_t_normal = self.sample(t=t, x_0=data, binomial=False).cpu()
        x_t_binomial = self.sample(t=t, x_0=data, binomial=False).cpu()

        norm_fn = lambda x: (x.clamp(-1, 1) + 1) / 2
        for x_t, name in zip([x_t_normal, x_t_binomial], ["Normal", "Binomial"]):
            if data.shape[1] == 1:
                fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
                idx = torch.round(torch.linspace(0, x_t.shape[0] - 1, 25)).int()
                for x_t_, t_, ax in zip(x_t[idx], t[idx], axs.flatten()):
                    tmp = ax.imshow(norm_fn(x_t_).squeeze(), vmin=0, vmax=1)
                    ax.set_title(f"t={t_.item():.2f} ({x_t_.min():.2f}/{x_t_.max():.2f})")
                    ax.set_xticks([]), ax.set_yticks([])
                _ = plt.colorbar(tmp)

            elif data.shape[1] == 3:
                fig = plt.figure()
                norm_fn = lambda x: (x.clamp(-1, 1) + 1) / 2
                x_t_ = norm_fn(x_t[:: x_t.shape[0] // min(64, x_t.shape[0])])
                grid = torchvision.utils.make_grid(x_t_).permute(1, 2, 0).cpu().numpy()
                print(f"{grid.min()=} {grid.max()=}")
                plt.imshow(grid)
                plt.suptitle(f"Data:{data.shape}; Every {x_t.shape[0] // 64}th image shown")

            plt.suptitle(f"{title} Perturbed Data T={self.T} ({name} sampling)")
            plt.tight_layout()
            plt.grid(False)
            wandb.log({f"NoiseSchedule/{title}PerturbedData ({name} sampling)": wandb.Image(fig)})
            plt.show()
            # sys.exit()
            # plt.close("all")
            # del fig

        if hasattr(self, "int_beta_t"):
            t = torch.linspace(0, self.T, 100)
            int_beta_t = self.int_beta_t(t)
            fig = plt.figure()
            plt.plot(t, int_beta_t)
            plt.title(f"{title} Rate (min:{int_beta_t.min().item():.2f})")
            wandb.log({f"NoiseSchedules/{title} Rate": wandb.Image(fig)})
            plt.show()
        # plt.close("all")
        # del fig

        if show:
            plt.show()
        plt.close("all")


if __name__ == "__main__":
    pass
