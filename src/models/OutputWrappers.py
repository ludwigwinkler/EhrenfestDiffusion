import torch


class Wrapper(torch.nn.Module):
	def __init__(self, model, model_config):
		super().__init__()
		self.model = model
		self.model_config = model_config


class BirthDeathOutputWrapper(Wrapper):

	def forward(self, x, t):
		assert x.min() >= -1. and x.max() <= 1.
		if self.model_config.residual:
			pred = torch.tanh(x + self.model(x=x, t=t))
		else:
			pred = torch.tanh(self.model(x=x, t=t))  # [-1,1]

		return pred


class VariationaDiscreteOutputWrapper(Wrapper):

	def forward(self, x, t):
		assert x.min() >= -1. and x.max() <= 1.
		pred = self.model(x=x, t=t)
		pred_mu, pred_logscale = pred.chunk(chunks=2, dim=1)
		if self.model_config.residual:
			pred_mu = torch.tanh(x + pred_mu)
		else:
			pred_mu = torch.tanh(pred_mu)  # [-1,1]

		pred_scale = torch.nn.functional.softplus(pred_logscale)
		dist = torch.distributions.Normal(loc=pred_mu, scale=pred_scale)
		return dist


def BirthDeathWrapperFn(model, x, t, residual):
	assert x.min() >= -1. and x.max() <= 1.
	if residual:
		pred = torch.tanh(x + model(x=x, t=t))
	else:
		pred = torch.tanh(model(x=x, t=t))  # [-1,1]
	return pred
