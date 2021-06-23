import itertools
from abc import ABC

import torch
import torch.nn as nn

import utils
from models.layers import *

__all__ = ['Autoencoder', 'FCAutoencoder', 'ConvAutoencoder']


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
        i = 0
        for child in itertools.chain(self._encoder.children(), self._decoder.children()):
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                if i == depth:
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
    def __init__(self, mode='A', apply_nested_dropout=False, optimize_dropout=False,
                 activation='ReLU', image_mode='RGB', **kwargs):
        super(ConvAutoencoder, self).__init__()
        if apply_nested_dropout:
            self._nested_dropout_layer = NestedDropout(**kwargs, optimize_dropout=optimize_dropout)
            self.optimize_dropout = optimize_dropout

        self.mode = mode
        self.apply_nested_dropout = apply_nested_dropout
        self._encoder, self._decoder = self.create_cae(image_mode, activation, **kwargs)

    def create_cae(self, image_mode, activation, normalize=True, batch_norm=False, **kwargs):
        """

        Parameters
        ----------
        image_mode - The format of the images the model receives [RGB/YCbCr/Y/...]
        activation - The activation function [ReLU/Tanh/...]
        normalize - Whether the images are normalized or not
        batch_norm - Whether batch norm should be applied
        kwargs - Catches unused key word arguments

        Returns
        -------
        Encoder and Decoder layers
        """
        activation_function = getattr(activations, activation)
        batch_norm = batch_norm
        channels = 1 if image_mode == 'Y' else 3

        if self.mode == 'A':
            channels_list = [64, 32, 16]
            conv_args_list = [{'kernel_size': 8, 'stride': 8},
                              {'kernel_size': 2, 'stride': 2},
                              {'kernel_size': 2, 'stride': 2}]
            encoder_layers, decoder_layers = self.generate_autoencoder_layers(channels_list, conv_args_list, channels,
                                                                              activation_function, batch_norm,
                                                                              normalize)

        elif self.mode == 'F':
            channels_list = [64, 128, 128]
            conv_args_list = [{'kernel_size': 8, 'stride': 8},
                              {'kernel_size': 2, 'stride': 2},
                              {'kernel_size': 2, 'stride': 2}]
            encoder_layers, decoder_layers = self.generate_autoencoder_layers(channels_list, conv_args_list, channels,
                                                                              activation_function, batch_norm,
                                                                              normalize)

        elif self.mode == 'G':
            channels_list = [64, 128, 256]
            conv_args_list = [{'kernel_size': 8, 'stride': 8},
                              {'kernel_size': 2, 'stride': 2},
                              {'kernel_size': 2, 'stride': 2}]
            encoder_layers, decoder_layers = self.generate_autoencoder_layers(channels_list, conv_args_list, channels,
                                                                              activation_function, batch_norm,
                                                                              normalize)

        elif self.mode == 'H':
            channels_list = [64, 64, 64]
            conv_args_list = [{'kernel_size': 8, 'stride': 8},
                              {'kernel_size': 2, 'stride': 2},
                              {'kernel_size': 2, 'stride': 2}]
            encoder_layers, decoder_layers = self.generate_autoencoder_layers(channels_list, conv_args_list, channels,
                                                                              activation_function, batch_norm,
                                                                              normalize)

        else:
            raise NotImplementedError(f'self.mode {self.mode} not implemented')

        return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)

    def generate_autoencoder_layers(self, channels_list, conv_args_list, channels,
                                    activation_function, batch_norm, normalized):
        encoder_layers = []
        decoder_layers = []
        first = True
        for new_channels, conv_args in zip(channels_list, conv_args_list):
            encoder_layers.append(nn.Conv2d(channels, new_channels, **conv_args))
            if batch_norm:
                encoder_layers.append(nn.BatchNorm2d(new_channels))
            encoder_layers.append(activation_function())

            if first:
                if self.apply_nested_dropout:
                    encoder_layers.append(self._nested_dropout_layer)
                first = False
                if not normalized:
                    # Scales our predictions to [0, 1] range
                    decoder_layers.append(nn.Sigmoid())
            else:
                decoder_layers.append(activation_function())
                if batch_norm:
                    decoder_layers.append(nn.BatchNorm2d(channels))
            decoder_layers.append(nn.ConvTranspose2d(new_channels, channels, **conv_args))

            channels = new_channels
        return encoder_layers, reversed(decoder_layers)

    def has_converged(self):
        if self.apply_nested_dropout:
            return self._nested_dropout_layer.has_converged()
        return None

    def get_converged_unit(self):
        if self.apply_nested_dropout:
            return self._nested_dropout_layer.converged_unit()
        return None

    def get_dropout_dim(self):
        if self.apply_nested_dropout:
            return self._nested_dropout_layer.dropout_dim()
        return None

    def get_feature_map(self, x: torch.Tensor, depth):
        out = x
        if out.dim() == 3:
            out = out.unsqueeze(0)

        i = 0
        for module in itertools.chain(self._encoder.children(), self._decoder.children()):
            out = module(out)
            if isinstance(module, nn.Conv2d):
                i += 1
                if i == depth:
                    break
        return out

    def forward_feature_map(self, feature_map: torch.Tensor, depth):
        out = feature_map
        if out.dim() == 3:
            out = out.unsqueeze(0)

        i = 0
        forward = False
        for module in itertools.chain(self._encoder.children(), self._decoder.children()):
            if forward:
                out = module(out)
            elif isinstance(module, nn.Conv2d):
                i += 1
                if i == depth:
                    forward = True
        return out
