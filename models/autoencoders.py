from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import linalg
from torch.distributions import Geometric

import utils


class Autoencoder(ABC, nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass


class FCAutoencoder(Autoencoder):
    def __init__(self, input_dim, representation_dim,
                 deep=False, activation=None, dropout=None):
        super(FCAutoencoder, self).__init__()
        self.repr_dim = representation_dim

        if deep:
            self._encoder = utils.create_sequential(input_dim, representation_dim,
                                                       activation=activation, dropout_p=dropout)
            self._decoder = utils.create_sequential(representation_dim, input_dim,
                                                       activation=activation, dropout_p=dropout)
        else:
            self._encoder = nn.Linear(input_dim, representation_dim)
            self._decoder = nn.Linear(representation_dim, input_dim)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        return self._encoder(x)

    def decode(self, x):
        return self._decoder(x)


class ConvAutoencoder(Autoencoder):
    def __init__(self, mode='dumb', activation='ReLU', filter_size=2):
        super(ConvAutoencoder, self).__init__()
        self._encoder, self._decoder = create_cae(activation, filter_size, mode)

        # if mode.startswith('VGG'):
        #     self._encoder, self._decoder = self.create_vgg(activation, mode)
        # else:
        #     raise NotImplementedError('Other modes of CAE have not yet been implemented')

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        for layer in self._encoder:
            x = layer(x)
        return x

    def decode(self, x):
        for layer in self._encoder:
            x = layer(x)
        return x


class NestedDropoutAutoencoder(Autoencoder):  # TODO implement
    def __init__(self, autoencoder: Autoencoder, input_dim, dropout_layer=None, tol=1e-3, sequence_bound=10,
                 distribution=Geometric, p=0.1):
        super(NestedDropoutAutoencoder, self).__init__()
        self._encoder = autoencoder._encoder
        self._decoder = autoencoder._decoder
        self.dropout_layer = dropout_layer
        self.input_dim = input_dim
        self.converged_unit = 0
        self.has_converged = False
        self.old_repr = None
        self.tol = tol
        self.sequence = 0
        self.max_sequence = sequence_bound
        self.distribution = distribution(p)

    def forward(self, x):
        x = self.encode(x)
        if self.dropout_layer is None and self.training:
            x = self.apply_nested_dropout(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        i = 0
        for layer in self._autoencoder.encoder:
            if self.dropout_layer is None:
                x = layer(x)
            else:
                if isinstance(layer, nn.Conv2d):
                    if i == self.dropout_layer:
                        x = self.apply_nested_dropout(x)
                    else:
                        i += 1
        return x

    def decode(self, x):
        return self._autoencoder.decode(x)

    def apply_nested_dropout(self, x):
        batch_size = x.shape[0]
        dropout_dim = x.shape[1]

        dropout_sample = self.distribution.sample((batch_size,)).type(torch.long)
        dropout_sample = torch.minimum(dropout_sample, dropout_dim)  # identical to above
        # dropout_sample[dropout_sample > (dropout_dim - 1)] = dropout_dim - 1

        mask = torch.tensor(torch.arange(dropout_dim) <= (dropout_sample.unsqueeze(1) + self.converged_unit)).to(x.device)
        return mask * x

    def check_convergence(self, x):
        new_repr = self.encode(x)

        difference = linalg.norm((new_repr - self.old_repr)[:, :self.converged_unit + 1]) / \
                     (len(x) * (self.converged_unit + 1))
        if difference <= self.tol:
            self.sequence += 1
        else:
            self.sequence = 0

        if self.sequence == self.max_sequence:
            self.sequence = 0
            self.converged_unit += 1

        if self.converged_unit == self.input_dim:
            self.has_converged = True


def create_cae(activation, filter_size, mode):
    activation_function = nn.Identity if activation is None else getattr(nn, activation)

    if mode == 'A':
        channels = 3
        dim = 32
        encoder_layers = []
        decoder_layers = []

        first = True
        while dim >= filter_size:
            new_channels = utils.get_power_successor(channels)
            encoder_layers.append(nn.Conv2d(channels, new_channels, filter_size, filter_size))
            encoder_layers.append(activation_function())

            if first:
                first = False
            else:
                decoder_layers.insert(0, activation_function())
            decoder_layers.insert(0, nn.ConvTranspose2d(new_channels, channels, filter_size, filter_size))

            channels = new_channels
            dim = dim / filter_size

    else:
        raise NotImplementedError()

    return encoder_layers, decoder_layers


def create_vgg_ae(activation, mode):
    def conv2d(in_ch, out_ch):
        return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def max_pool2d():
        return nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def conv_trans2d(in_ch, out_ch):
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=1)

    def max_unpool2d():
        return nn.MaxUnpool2d(kernel_size=2, stride=2)

    vgg_variant = mode.split('_')[1]

    if vgg_variant != 'A':
        raise NotImplementedError('Only the A variant of vgg has been implemented as of yet')
    # TODO implement more variants if needed

    if activation is None:
        act = nn.Identity
    else:
        act = getattr(nn, activation)

    encoder_modules = [conv2d(3, 64), act(),
                       max_pool2d(),
                       conv2d(64, 128), act(),
                       max_pool2d(),
                       conv2d(128, 256), act(),
                       conv2d(256, 256), act(),
                       max_pool2d(),
                       conv2d(256, 512), act(),
                       conv2d(512, 512), act(),
                       max_pool2d(),
                       conv2d(512, 512), act(),
                       conv2d(512, 512), act(),
                       max_pool2d()]
    decoder_models = [max_unpool2d(),
                      conv_trans2d(512, 512), act(),
                      conv_trans2d(512, 512), act(),
                      max_unpool2d(),
                      conv_trans2d(512, 512), act(),
                      conv_trans2d(512, 256), act(),
                      max_unpool2d(),
                      conv_trans2d(256, 256), act(),
                      conv_trans2d(256, 128), act(),
                      max_unpool2d(),
                      conv_trans2d(128, 64), act(),
                      max_unpool2d(),
                      conv_trans2d(64, 3), act()]

    return nn.Sequential(*encoder_modules), \
           nn.Sequential(*decoder_models)
