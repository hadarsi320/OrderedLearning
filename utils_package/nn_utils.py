import torch.nn as nn
from math import log2, ceil, floor


def create_sequential(start_dim: int, end_dim: int, activation: str = None):
    assert start_dim != end_dim

    dimensions = []
    if start_dim > end_dim:
        hidden_start = floor(log2(start_dim))
        hidden_end = ceil(log2(end_dim))
        for i in range(hidden_start, hidden_end-1, -1):
            dimensions.append(2 ** i)
    else:
        hidden_start = ceil(log2(start_dim))
        hidden_end = floor(log2(end_dim))
        for i in range(hidden_start, hidden_end+1, 1):
            dimensions.append(2 ** i)

    layers = []
    last_dim = start_dim
    for dim in dimensions:
        layers.append(nn.Linear(last_dim, dim))
        if activation is not None:
            layers.append(getattr(nn, activation)())
    layers.append(nn.Linear(last_dim, dim))

    return nn.Sequential(*layers)


if __name__ == '__main__':
    print(create_sequential(100, 3072, activation='ReLU'))
    print(create_sequential(3072, 100, activation='Sigmoid'))
