import torch
from torch import nn, linalg
from torch.distributions import Geometric

import utils

__all__ = ['NestedDropout']


class NestedDropout(nn.Module):
    def __init__(self, tol=1e-3, sequence_bound=10, distribution=Geometric(0.1)):
        # TODO set default parameters
        super(NestedDropout, self).__init__()
        self.tol = tol
        self.sequence = 0
        self.sequence_bound = sequence_bound
        self.distribution = distribution
        self.dropout_dim = None
        self.converged_unit = 0
        self.has_converged = False
        self.old_repr = None

    def forward(self, x):
        if self.training:
            batch_size = x.shape[0]
            if self.dropout_dim is None:
                self.dropout_dim = x.shape[1]

            dropout_sample = self.distribution.sample((batch_size,)).type(torch.long)
            dropout_sample = torch.minimum(dropout_sample, torch.tensor(self.dropout_dim - 1))  # identical to above

            mask = (torch.arange(self.dropout_dim) <= (dropout_sample.unsqueeze(1) + self.converged_unit)) \
                .to(x.device)
            mask = utils.fit_dim(mask, x)
            x = mask * x

            if self.old_repr is None:
                self.old_repr = x
            else:
                self.check_convergence(x)
                self.old_repr = None
        return x

    def check_convergence(self, x):
        difference = (linalg.norm((x - self.old_repr)[:, :self.converged_unit + 1]) /
                      (len(x) * (self.converged_unit + 1)))
        if difference <= self.tol:
            self.sequence += 1
        else:
            self.sequence = 0

        if self.sequence == self.sequence_bound:
            self.sequence = 0
            self.converged_unit += 1

        if self.converged_unit == self.dropout_dim:
            self.has_converged = True
