import torch
from torch import nn

from models.layers import *

__all__ = ['Classifier']


class Classifier(nn.Module):
    def __init__(self, mode: str, num_classes: int, apply_nested_dropout: bool = False, **kwargs):
        super(Classifier, self).__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.apply_nested_dropout = apply_nested_dropout
        self._classifier = self.__create_classifier(**kwargs)

    def __create_classifier(self, **kwargs):
        act_function = getattr(nn, kwargs.get('activation', 'ReLU'))
        batch_norm = kwargs.get('batch_norm', True)
        channels = kwargs.get('channels', 3)

        if self.mode == 'A':
            channels_list = [32, 64, 64, 128]
            conv_args_list = [{'kernel_size': 8, 'stride': 8},
                              {'kernel_size': 2, 'stride': 2},
                              {'kernel_size': 2, 'stride': 2},
                              {'kernel_size': 7, 'stride': 7}]
            dims_list = [100, self.num_classes]
            layers = self.__generate_layers(channels_list, conv_args_list, dims_list,
                                            **kwargs)

        elif self.mode == 'B':
            channels_list = [64, 64]
            conv_args_list = [{'kernel_size': 8, 'stride': 8},
                              {'kernel_size': 3, 'padding': 1}]

            for curr_channels in [128, 256]:
                for i in range(2):
                    channels_list.append(curr_channels)
                    conv_args = {'kernel_size': 3, 'padding': 1}
                    if i == 0:
                        conv_args['stride'] = 2
                    conv_args_list.append(conv_args)

            dims_list = [1000, self.num_classes]
            average_pool = nn.AvgPool2d(kernel_size=7)
            layers = self.__generate_layers(channels_list, conv_args_list, dims_list, **kwargs,
                                            final_pooling=average_pool)

        elif self.mode == 'ResNet34':
            planes = 64
            layers = [nn.Conv2d(channels, planes, kernel_size=8, stride=8),
                      # nn.BatchNorm2d(64),
                      # act_function(),
                      ]
            if self.apply_nested_dropout:
                self._dropout_layer = NestedDropout(**kwargs)
                layers.append(self._dropout_layer)

            for reps in [4, 6, 3]:
                out_planes = planes * 2
                for i in range(reps):
                    down_sampling = i == 0 and reps != 4
                    layers.append(BasicBlock(planes, out_planes, activation_layer=act_function,
                                             batch_norm=batch_norm, downsampling=down_sampling))
                    planes = out_planes

            layers.append(nn.AvgPool2d(kernel_size=7))
            layers.append(nn.Flatten(-3))
            layers.append(nn.Linear(planes, 1000))
            layers.append(nn.Linear(1000, self.num_classes))

        elif self.mode == 'ResNet50':
            pass

        else:
            raise NotImplementedError(f'Mode {self.mode} not implemented')

        sequential = nn.Sequential(*layers)
        return sequential

    def __generate_layers(self, channels_list, conv_args_list, dims_list, **kwargs):
        act_function = getattr(nn, kwargs.get('activation', 'ReLU'))
        batch_norm = kwargs.get('batch_norm', True)
        channels = kwargs.get('channels', 3)
        apply_nested_dropout = self.apply_nested_dropout

        layers = []
        last_channels = channels
        for channels, conv_args in zip(channels_list, conv_args_list):
            layers.append(nn.Conv2d(last_channels, channels, **conv_args))
            if batch_norm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(act_function())

            if apply_nested_dropout:
                self._dropout_layer = NestedDropout(**kwargs)
                layers.append(self._dropout_layer)
                apply_nested_dropout = False
            last_channels = channels

        if 'final_pooling' in kwargs:
            layers.append(kwargs['final_pooling'])
        layers.append(nn.Flatten(-3))

        last_dim = channels_list[-1]
        for i, dim in enumerate(dims_list):
            layers.append(nn.Linear(last_dim, dim))
            if i + 1 < len(dims_list):
                layers.append(act_function())
                # if batch_norm:
                #     layers.append(nn.BatchNorm1d(dim))
            last_dim = dim
        return layers

    def forward(self, x):
        x = self._classifier(x)
        return x

    def predict(self, x):
        x = self(x)
        x = torch.argmax(x, -1)
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

    def get_weights(self, depth):
        i = 0
        for child in self._classifier.children():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                i += 1
                if i == depth:
                    return child.weight, child.bias

        raise ValueError('Depth is too deep')
