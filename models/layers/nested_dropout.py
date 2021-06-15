import torch
from torch import nn, linalg
from torch.distributions import Geometric

import utils

__all__ = ['NestedDropout']


# TODO create an __str__ function for this module
class NestedDropout(nn.Module):
    def __init__(self, p=0.1, optimize_dropout=True, tol=1e-3, sequence_bound=2 ** 4, **kwargs):
        super(NestedDropout, self).__init__()
        self.distribution = Geometric(p)
        self.optimize_dropout = optimize_dropout
        self._dropout_dim = None
        self._converged_unit = 0
        self._has_converged = False

        if self.optimize_dropout:
            self._tol = tol
            self._freeze_gradients = freeze_gradients
            self._sequence = 0
            self._sequence_bound = sequence_bound
            self._old_repr = None

    def forward(self, x):
        if self.training and not self._has_converged:
            if self.optimize_dropout and self._old_repr is not None:
                self.check_convergence(x)
                self._old_repr = None
                return x.detach()
                # detaching so no gradients flow when checking convergence

            self._old_repr = x
            batch_size = x.shape[0]
            if self._dropout_dim is None:
                self._dropout_dim = x.shape[1]

            dropout_sample = self.distribution.sample((batch_size,)).type(torch.long)
            dropout_sample = torch.minimum(dropout_sample, torch.tensor(self._dropout_dim - 1))

            mask = (torch.arange(self._dropout_dim) <= (dropout_sample.unsqueeze(1) + self._converged_unit))
            mask = utils.fit_dim(mask, x).to(x.device)
            x = mask * x

            if self.optimize_dropout and self._freeze_gradients:
                x[:, :self._converged_unit] = x[:, :self._converged_unit].detach()
        return x

    def check_convergence(self, x):
        # z = (x - self._old_repr)[:, :self._converged_unit + 1].reshape(-1)
        z = (x - self._old_repr)[:, self._converged_unit].reshape(-1)
        difference = linalg.norm(z) / len(z)

        if difference <= self._tol:
            self._sequence += 1
            if self._sequence == self._sequence_bound:
                self._sequence = 0
                self._converged_unit += 1
                if self._converged_unit == self._dropout_dim:
                    self._has_converged = True
        else:
            self._sequence = 0

    def has_converged(self):
        return self._has_converged

    def converged_unit(self):
        return self._converged_unit

    def dropout_dim(self):
        return self._dropout_dim
