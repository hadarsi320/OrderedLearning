import torch.nn as nn

from utils import nn_utils


class Autoencoder(nn.Module):
    def __init__(self, input_dim, representation_dim,
                 deep=False, activation=None, dropout=None):
        super(Autoencoder, self).__init__()
        self.repr_dim = representation_dim

        if deep:
            self._encoder = nn_utils.create_sequential(input_dim, representation_dim,
                                                       activation=activation, dropout_p=dropout)
            self._decoder = nn_utils.create_sequential(representation_dim, input_dim,
                                                       activation=activation, dropout_p=dropout)
        else:
            self._encoder = nn.Linear(input_dim, representation_dim)
            self._decoder = nn.Linear(representation_dim, input_dim)

    def forward(self, x):
        x = self._encoder(x)
        return self._decoder(x)

    def encode(self, x):
        return self._encoder(x)

    def decode(self, x):
        return self._decoder(x)
