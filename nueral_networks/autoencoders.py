import torch
import torch.nn as nn
from torch.distributions import Geometric

from utils_package import nn_utils


class Autoencoder(nn.Module):
    def __init__(self, input_dim, representation_dim,
                 apply_nested_dropout=False, eps=1e-2, deep=False, activation=None):
        super(Autoencoder, self).__init__()
        self.conv_i = 0
        self.converged = False
        self.last_repr = None
        self.conv_succession = 0
        self.eps = eps
        self.repr_dim = representation_dim

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if deep:
            self.encoder = nn_utils.create_sequential(input_dim, representation_dim, activation=activation)
            self.decoder = nn_utils.create_sequential(representation_dim, input_dim, activation=activation)
        else:
            self.encoder = nn.Linear(input_dim, representation_dim).to(self.device)
            self.decoder = nn.Linear(representation_dim, input_dim).to(self.device)

        self.apply_nested_dropout = apply_nested_dropout

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder(x)
        if self.apply_nested_dropout and self.training is True and not self.converged:
            x = self.nested_dropout(x, self.conv_i)
            self.last_repr = x
        return self.decoder(x)

    def get_representation(self, x):
        x = x.to(self.device)
        return self.encoder(x)

    def get_reconstructions(self, x):
        x = x.to(self.device)
        return self.decoder(x)

    def check_convergence(self, x):
        if not self.converged:
            x_repr = self.get_representation(x)
            diff = torch.norm(self.last_repr[:, self.conv_i] - x_repr[:, self.conv_i]) / x_repr.shape[0]
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

    def nested_dropout(self, x, min_neuron, p=0.1):
        batch_size = x.shape[0]
        neurons = x.shape[1]

        dist = Geometric(probs=p)
        dropout_sample = dist.sample((batch_size,)).type(torch.long)
        dropout_sample[dropout_sample > (neurons - 1)] = neurons - 1

        mask = (torch.arange(neurons) <= (dropout_sample.unsqueeze(1) + min_neuron)).to(self.device)
        return mask * x
