import itertools
from abc import ABC

import torch
import torch.nn as nn

import utils
from models.layers import *

__all__ = ['Autoencoder', 'FCAutoencoder', 'ConvAutoencoder', 'NestedDropoutAutoencoder']


class Autoencoder(ABC, nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self._encoder(x)
        return x

    def decode(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self._decoder(x)
        return x

    def get_weights(self, depth):
        i = 1
        for child in itertools.chain(self._encoder.children(), self._decoder.children()):
            if isinstance(child, (nn.Conv2d, nn.Linear)) and i == depth:
                return child.weight, child.bias
            i += 1

        raise ValueError('Depth is too deep')


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


class ConvAutoencoder(Autoencoder):
    def __init__(self, mode='A', dim=32, **kwargs):
        super(ConvAutoencoder, self).__init__()
        self._encoder, self._decoder = create_cae(mode, dim, **kwargs)


class NestedDropoutAutoencoder(Autoencoder):
    def __init__(self, autoencoder: Autoencoder, dropout_depth=None, **kwargs):
        super(NestedDropoutAutoencoder, self).__init__()
        self._encoder = autoencoder._encoder
        self._decoder = autoencoder._decoder
        self._dropout_depth = dropout_depth
        self._dropout_layer = self.__split_encoder(**kwargs)

    def __split_encoder(self, **kwargs):
        nested_dropout_layer = NestedDropout(**kwargs)
        encoder_layers = list(self._encoder.children())
        decoder_layers = list(self._decoder.children())
        if self._dropout_depth is not None:
            i = 0
            for j, layer in enumerate(encoder_layers):
                if isinstance(layer, nn.Conv2d):
                    i += 1
                    if i == self._dropout_depth:
                        self._encoder = nn.Sequential(*encoder_layers[:j + 1])
                        self._decoder = nn.Sequential(*(encoder_layers[j + 1:] + decoder_layers))
                        i = None
                        break
            if i is not None:
                raise ValueError('Dropout depth is too deep')

        return nested_dropout_layer

    def forward(self, x):
        x = self._encoder(x)
        x = self._dropout_layer(x)
        x = self._decoder(x)
        return x

    def has_converged(self):
        return self._dropout_layer.has_converged

    def get_converged_unit(self):
        return self._dropout_layer.converged_unit

    def get_dropout_dim(self):
        return self._dropout_layer.dropout_dim


def create_cae(mode, starting_dim, batch_norm=True, **kwargs):
    activation_function = getattr(nn, kwargs.get('activation', 'ReLU'))
    normalized = kwargs.get('normalize_data', True)

    channels = kwargs.get('channels', 3)
    dim = starting_dim
    encoder_layers = []
    decoder_layers = []

    if mode == 'A':
        filter_size = 2
        first = True
        while dim >= filter_size:
            new_channels = utils.get_power_successor(channels)
            encoder_layers.append(nn.Conv2d(channels, new_channels, filter_size, filter_size))
            encoder_layers.append(activation_function())
            if batch_norm:
                encoder_layers.append(nn.BatchNorm2d(new_channels))

            if first:
                first = False
                if not normalized:
                    decoder_layers.append(nn.Sigmoid())  # Scaled our predictions to [0, 1] range
            else:
                if batch_norm:
                    decoder_layers.append(nn.BatchNorm2d(channels))
                decoder_layers.append(activation_function())
            decoder_layers.append(nn.ConvTranspose2d(new_channels, channels, filter_size, filter_size))

            channels = new_channels
            dim = dim / filter_size
        decoder_layers = reversed(decoder_layers)

    elif mode == 'B':
        channels = 3
        dim = 32
        encoder_layers = []
        decoder_layers = []

        first = True
        while dim >= 2:
            new_channels = utils.get_power_successor(channels)
            encoder_layers.append(nn.Conv2d(channels, new_channels, 4, stride=2, padding=1))
            encoder_layers.append(activation_function())
            if batch_norm:
                encoder_layers.append(nn.BatchNorm2d(new_channels))

            if first:
                first = False
                if not normalized:
                    decoder_layers.append(nn.Sigmoid())  # Scaled our predictions to [0, 1] range
            else:
                if batch_norm:
                    decoder_layers.insert(0, nn.BatchNorm2d(channels))
                decoder_layers.insert(0, activation_function())
            decoder_layers.insert(0, nn.ConvTranspose2d(new_channels, channels, 4, stride=2, padding=1))

            channels = new_channels
            dim = dim / 2

    elif mode == 'C':
        channels_list = [32, 32, 64, 64]

        encoder_layers = []
        decoder_layers = []
        first = True
        for new_channels in channels_list:
            encoder_layers.append(nn.Conv2d(channels, new_channels, 4, stride=2, padding=1))
            encoder_layers.append(activation_function())
            if batch_norm:
                encoder_layers.append(nn.BatchNorm2d(new_channels))

            if first:
                first = False
                if not normalized:
                    decoder_layers.append(nn.Sigmoid())  # Scaled our predictions to [0, 1] range
            else:
                if batch_norm:
                    decoder_layers.insert(0, nn.BatchNorm2d(channels))
                decoder_layers.insert(0, activation_function())
            decoder_layers.insert(0, nn.ConvTranspose2d(new_channels, channels, 4, stride=2, padding=1))

            channels = new_channels
            dim = dim / 2

    elif mode == 'D':
        channels_list = [8]
        conv_args = {'kernel_size': 8, 'stride': 8}

        encoder_layers = []
        decoder_layers = []
        first = True
        for new_channels in channels_list:
            encoder_layers.append(nn.Conv2d(channels, new_channels, **conv_args))
            encoder_layers.append(activation_function())
            if batch_norm:
                encoder_layers.append(nn.BatchNorm2d(new_channels))

            if first:
                first = False
                if not normalized:
                    decoder_layers.append(nn.Sigmoid())  # Scaled our predictions to [0, 1] range
            else:
                if batch_norm:
                    decoder_layers.insert(0, nn.BatchNorm2d(channels))
                decoder_layers.insert(0, activation_function())
            decoder_layers.insert(0, nn.ConvTranspose2d(new_channels, channels, **conv_args))

            channels = new_channels
            dim = dim / 2

    elif mode == 'E':
        channels_list = [64]
        conv_args = {'kernel_size': 8, 'stride': 8}

        encoder_layers = []
        decoder_layers = []
        first = True
        for new_channels in channels_list:
            encoder_layers.append(nn.Conv2d(channels, new_channels, **conv_args))
            encoder_layers.append(activation_function())
            if batch_norm:
                encoder_layers.append(nn.BatchNorm2d(new_channels))

            if first:
                first = False
                if not normalized:
                    decoder_layers.append(nn.Sigmoid())  # Scaled our predictions to [0, 1] range
            else:
                if batch_norm:
                    decoder_layers.insert(0, nn.BatchNorm2d(channels))
                decoder_layers.insert(0, activation_function())
            decoder_layers.insert(0, nn.ConvTranspose2d(new_channels, channels, **conv_args))

            channels = new_channels
            dim = dim / 2

    elif mode == 'F':
        channels_list = [64, 64, 128]
        conv_args_list = [{'kernel_size': 8, 'stride': 8},
                     {'kernel_size': 2, 'stride': 2},
                     {'kernel_size': 2, 'stride': 2}]

        encoder_layers = []
        decoder_layers = []
        first = True
        for new_channels, conv_args in zip(channels_list, conv_args_list):
            encoder_layers.append(nn.Conv2d(channels, new_channels, **conv_args))
            encoder_layers.append(activation_function())
            if batch_norm:
                encoder_layers.append(nn.BatchNorm2d(new_channels))

            if first:
                first = False
                if not normalized:
                    decoder_layers.append(nn.Sigmoid())  # Scaled our predictions to [0, 1] range
            else:
                if batch_norm:
                    decoder_layers.insert(0, nn.BatchNorm2d(channels))
                decoder_layers.insert(0, activation_function())
            decoder_layers.insert(0, nn.ConvTranspose2d(new_channels, channels, **conv_args))

            channels = new_channels
    else:
        raise NotImplementedError()

    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)
