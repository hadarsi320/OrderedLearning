from abc import ABC

import torch.nn as nn

import utils
from .layers import *

__all__ = ['Autoencoder', 'FCAutoencoder', 'ConvAutoencoder', 'NestedDropoutAutoencoder']


class Autoencoder(ABC, nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        pass

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self._encoder(x)
        return x

    def decode(self, x):
        x = self._decoder(x)
        return x


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
    def __init__(self, mode='A', activation='ReLU', filter_size=2):
        super(ConvAutoencoder, self).__init__()
        self._encoder, self._decoder = create_cae(activation, filter_size, mode)

        # if mode.startswith('VGG'):
        #     self._encoder, self._decoder = self.create_vgg(activation, mode)
        # else:
        #     raise NotImplementedError('Other modes of CAE have not yet been implemented')


class NestedDropoutAutoencoder(Autoencoder):
    def __init__(self, autoencoder: Autoencoder, dropout_depth=None, **kwargs):
        super(NestedDropoutAutoencoder, self).__init__()
        self._encoder = autoencoder._encoder
        self._decoder = autoencoder._decoder
        self._dropout_layer = self.add_nested_dropout(dropout_depth, **kwargs)

    def add_nested_dropout(self, dropout_depth, **kwargs):
        nested_dropout_layer = NestedDropout(**kwargs)
        layers = list(self._encoder.children())
        if dropout_depth is None:
            layers.append(nested_dropout_layer)
        else:
            i = 0
            for j, layer in enumerate(layers):
                if isinstance(layer, nn.Conv2d):
                    i += 1
                    if i == dropout_depth:
                        layers.insert(j + 1, nested_dropout_layer)
                        break

        self._encoder = nn.Sequential(*layers)
        return nested_dropout_layer

    def has_converged(self):
        return self._dropout_layer.has_converged

    def get_converged_unit(self):
        return self._dropout_layer.converged_unit

    def get_dropout_dim(self):
        return self._dropout_layer.dropout_dim


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

    # return encoder_layers, decoder_layers
    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


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
