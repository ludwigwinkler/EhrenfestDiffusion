import torch
import torch.nn.functional as F


class BasePredictorClass(torch.nn.Module):
    def __init__(self, output_type="gaussian"):
        super().__init__()
        self.output_type = output_type

    def forward(self, x, t):
        # assert x.min() >= -1.0 and x.max() <= 1.0

        '''Call to the underlying neural network model'''
        output: torch.Tensor = self.predict(x=x, t=t)

        '''Transform neural network output into desired quantity'''
        if self.output_type in ['epsilon', 'score', 'taylor1']:
            output_dict = {self.output_type: output, 'output_type': self.output_type, 'output': output}
        elif self.output_type == 'ratio':
            output = torch.exp(output)  # Gaussian Approximation Rates are exp(linear_function), so we do the same
            output_dict = {'ratio': output, 'x_0_t': x, 'output_type': 'ratio',
                           'output': output.detach()}  # x_0_t is best estimate
        elif self.output_type == 'ratio2':
            # Gaussian Approximation Rates are exp(linear_function), so we do the same
            output_dict = {'ratio2': output, 'x_0_t': x, 'output': output,
                           'output_type': self.output_type}  # x_0_t is best estimate
        else:
            raise NotImplementedError(f"DiffuserNet: {self.output_type}")
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
            # raise NotImplementedError(f"DiffuserNet: {self.output_type}")
