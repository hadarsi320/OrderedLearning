import torch
from torch import nn

from models.layers import *


class Classifier(nn.Module):
    def __init__(self, mode: str, num_classes: int, apply_nested_dropout: bool, **kwargs):
        super(Classifier, self).__init__()
        self._mode = mode
        self._num_classes = num_classes
        self.apply_nested_dropout = apply_nested_dropout
        self._classifier = self.__create_classifier(mode, num_classes, **kwargs)

    def __create_classifier(self, mode, num_classes, **kwargs):
        activation_function = getattr(nn, kwargs.get('activation', 'ReLU'))
        batch_norm = kwargs.get('batch_norm', True)
        channels = kwargs.get('channels', 3)
        apply_nested_dropout = self.apply_nested_dropout

        layers = []
        if mode == 'A':
            channels_list = [32, 64, 64, 128]
            conv_args_list = [{'kernel_size': 8, 'stride': 8},
                              {'kernel_size': 2, 'stride': 2},
                              {'kernel_size': 2, 'stride': 2},
                              {'kernel_size': 7, 'stride': 7}]
            dims = [100, num_classes]

            last_channels = channels
            for channels, conv_args in zip(channels_list, conv_args_list):
                layers.append(nn.Conv2d(last_channels, channels, **conv_args))
                layers.append(activation_function())
                if batch_norm:
                    layers.append(nn.BatchNorm2d(channels))
                if apply_nested_dropout:
                    self.nested_dropout_layer = NestedDropout(**kwargs)
                    layers.append(self.nested_dropout_layer)
                    apply_nested_dropout = False
                last_channels = channels

            layers.append(nn.Flatten(-3))

            last_dim = channels_list[-1]
            for i, dim in enumerate(dims):
                layers.append(nn.Linear(last_dim, dim))
                if i + 1 < len(dims):
                    layers.append(activation_function())
                    # if batch_norm:
                    #     layers.append(nn.BatchNorm1d(dim))
                last_dim = dim

        else:
            raise NotImplementedError(f'Mode {mode} not implemented')

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self._classifier(x)
        return x

    def predict(self, x):
        x = self(x)
        x = torch.softmax(x, -1)
        return x

    def has_converged(self):
        if not self.apply_nested_dropout:
            return None
        return self._dropout_layer.has_converged

    def get_converged_unit(self):
        if not self.apply_nested_dropout:
            return None
        return self._dropout_layer.converged_unit

    def get_dropout_dim(self):
        if not self.apply_nested_dropout:
            return None
        return self._dropout_layer.dropout_dim
