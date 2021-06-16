import torch
from torch import nn, linalg
from torch.distributions import Geometric

import utils

__all__ = ['NestedDropout']


# TODO create an __str__ function for this module
class NestedDropout(nn.Module):
    def __init__(self, p=0.1, tol=1e-3, freeze_gradients=False, past_module: nn.Module = None, **kwargs):
        super(NestedDropout, self).__init__()
        self.distribution = Geometric(p)
        self.past_module = past_module
        self._freeze_gradients = freeze_gradients
        self._tol = tol
        self._converged_unit = 0
        self._dropout_dim = None
        self._has_converged = False
        self._old_weight = None

    def forward(self, x):
        if self._dropout_dim is None:
            self._dropout_dim = x.shape[1]

        if self.training and not self._has_converged:
            x = nested_dropout(x, self.distribution, self._converged_unit)
            if self._freeze_gradients:
                x[:, :self._converged_unit] = x[:, :self._converged_unit].detach()
        return x

    def save_weight(self):
        self._old_weight = torch.clone(self.past_module.weight).detach()

    def check_convergence(self):
        if self._old_weight is None:
            raise Exception('check_convergence ran before save_weight ran')

        z = (self.past_module.weight - self._old_weight)[self._converged_unit].reshape(-1)
        difference = linalg.norm(z) / len(z)

        if difference <= self._tol:
            self._converged_unit += 1
            if self._converged_unit == self._dropout_dim:
                self._has_converged = True
        self._old_weight = None

    def has_converged(self):
        return self._has_converged

    def converged_unit(self):
        return self._converged_unit

    def dropout_dim(self):
        return self._dropout_dim

    def __str__(self):
        return f'Nested Dropout distribution={self.distribution}'


def nested_dropout(x: torch.Tensor, distribution=None, converged_unit=0, p=0.1):
    if distribution is None:
        distribution = Geometric(p)

    batch_size = x.shape[0]
    dropout_dim = x.shape[1]
    dropout_sample = distribution.sample((batch_size,)).type(torch.long)
    mask = (torch.arange(dropout_dim) <= (dropout_sample.unsqueeze(1) + converged_unit))
    mask = utils.fit_dim(mask, x).to(x.device)
    x = mask * x
    return x
