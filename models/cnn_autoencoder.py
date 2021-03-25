import torch.nn as nn

from utils import nn_utils


class CAE(nn.Module):
    def __init__(self, input_dim, representation_dim,
                 mode='VGG', activation=None, dropout=True):
        super(CAE, self).__init__()
        self.repr_dim = representation_dim

        if mode == 'VGG':
            self._encoder = self.create_conv_sequential(input_dim)
            self._decoder = self.create_conv_sequential(input_dim)
        else:
            raise NotImplementedError('Other modes of CAE have not yet been implemented')

    def create_conv_sequential(self):
        pass

    def forward(self, x):
        x = self._encoder(x)
        return self._decoder(x)

    def encode(self, x):
        return self._encoder(x)

    def decode(self, x):
        return self._decoder(x)
