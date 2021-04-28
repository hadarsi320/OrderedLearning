import torch
from torch import nn, linalg
from torch.distributions import Geometric

import utils

__all__ = ['NestedDropout']


class NestedDropout(nn.Module):
    def __init__(self, tol=1e-3, sequence_bound=10, p=0.1, **kwargs):
        super(NestedDropout, self).__init__()
        self.tol = tol
        self.sequence = 0
        self.sequence_bound = sequence_bound
        self.distribution = Geometric(p)
        self.dropout_dim = None
        self.converged_unit = 0
        self.has_converged = False
        self.old_repr = None

    def forward(self, x):
        if self.training and not self.has_converged:
            batch_size = x.shape[0]
            if self.dropout_dim is None:
                self.dropout_dim = x.shape[1]

            dropout_sample = self.distribution.sample((batch_size,)).type(torch.long)
            dropout_sample = torch.minimum(dropout_sample, torch.tensor(self.dropout_dim - 1))

            mask = (torch.arange(self.dropout_dim) <= (dropout_sample.unsqueeze(1) + self.converged_unit))
            mask = utils.fit_dim(mask, x).to(x.device)
            x = mask * x

            if self.old_repr is None:
                self.old_repr = x
            else:
                self.check_convergence(x)
                self.old_repr = None
        return x

    def check_convergence(self, x):
        z = (x - self.old_repr)[:, :self.converged_unit + 1].reshape(-1)
        difference = linalg.norm(z) / len(z)

        if difference <= self.tol:
            self.sequence += 1
            if self.sequence == self.sequence_bound:
                self.sequence = 0
                self.converged_unit += 1
                if self.converged_unit == self.dropout_dim:
                    self.has_converged = True
        else:
            self.sequence = 0
