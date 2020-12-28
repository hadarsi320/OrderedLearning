import torch
import torch.nn as nn
from torch.distributions import Geometric


def nested_dropout(x: torch.Tensor, min_neuron, p=0.1):
    batch_size = x.shape[0]
    neurons = x.shape[1]

    dist = Geometric(probs=p)
    dropout_sample = dist.sample((batch_size,)).type(torch.long)
    dropout_sample[dropout_sample > (neurons - 1)] = neurons - 1

    mask = torch.arange(neurons) <= (dropout_sample.unsqueeze(1) + min_neuron)
    return mask * x


class Autoencoder(nn.Module):
    def __init__(self, input_dim, representation_dim, apply_nested_dropout=False, eps=1e-2):
        super(Autoencoder, self).__init__()
        self.conv_i = 0
        self.converged = False
        self.last_repr = None
        self.conv_succession = 0
        self.eps = eps
        self.repr_dim = representation_dim

        self.encoder = nn.Linear(input_dim, representation_dim)
        self.decoder = nn.Linear(representation_dim, input_dim)
        self.apply_nested_dropout = apply_nested_dropout

    def forward(self, x):
        x = self.encoder(x)
        if self.apply_nested_dropout and self.training is True and not self.converged:
            x = nested_dropout(x, self.conv_i)
            self.last_repr = x
        return self.decoder(x)

    def get_repr(self, x):
        return self.encoder(x)

    def check_convergence(self, x):
        if not self.converged:
            repr = self.get_repr(x)
            diff = torch.norm(self.last_repr[:, self.conv_i] - repr[:, self.conv_i]) / repr.shape[0]
            if diff <= self.eps:
                self.conv_succession += 1
            else:
                self.conv_succession = 0

            if self.conv_succession == 20:
                if self.conv_i == self.repr_dim - 1:
                    self.converged = True
                else:
                    self.conv_i += 1
                    self.conv_succession = 0

