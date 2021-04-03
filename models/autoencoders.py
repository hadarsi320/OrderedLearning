from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from utils import nn_utils


class Autoencoder(ABC, nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        pass

    @abstractmethod
    def forward(self, x):
        x = self._encoder(x)
        return self._decoder(x)

    @abstractmethod
    def encode(self, x):
        return self._encoder(x)

    @abstractmethod
    def decode(self, x):
        return self._decoder(x)


class FullyConnectedAutoencoder(Autoencoder):
    def __init__(self, input_dim, representation_dim,
                 deep=False, activation=None, dropout=None):
        super(FullyConnectedAutoencoder, self).__init__()
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


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_dim=(224, 224, 3), representation_dim=(49, 49, 512),
                 mode='VGG_A', activation='ReLU', dropout=True):
        super(ConvolutionalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.representation_dim = representation_dim

        if self.input_dim != (224, 224, 3) or self.representation_dim != (49, 49, 512):
            raise NotImplementedError('Convolutional Autoencoder with different dimensions is yet to be implemented')

        if mode.startswith('VGG'):
            self._encoder, self._decoder = self.create_vgg(activation, mode)
        else:
            raise NotImplementedError('Other modes of CAE have not yet been implemented')

    def create_vgg(self, activation, mode):
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

    def forward(self, x):
        x = self._encoder(x)
        return self._decoder(x)

    def encode(self, x):
        return self._encoder(x)

    def decode(self, x):
        return self._decoder(x)


class NestedDropoutAutoencoder():  # TODO implement
    pass


if __name__ == '__main__':
    autoencoder = ConvolutionalAutoencoder()
    tensor = torch.rand(16, 3, 224, 224)
    autoencoder(tensor)
    enc = autoencoder.encode(tensor)
    dec = autoencoder.decode(tensor)
