from math import log2, ceil, floor

import torch.nn as nn


def create_sequential(start_dim: int, end_dim: int, activation: str = None):
    assert start_dim != end_dim

    dimensions = []
    if start_dim > end_dim:
        hidden_start = floor(log2(start_dim))
        hidden_end = ceil(log2(end_dim))
        for i in range(hidden_start, hidden_end - 1, -1):
            dimensions.append(2 ** i)
    else:
        hidden_start = ceil(log2(start_dim))
        hidden_end = floor(log2(end_dim))
        for i in range(hidden_start, hidden_end + 1, 1):
            dimensions.append(2 ** i)

    layers = []
    last_dim = start_dim
    for dim in dimensions:
        layers.append(nn.Linear(last_dim, dim))
        if activation is not None:
            layers.append(getattr(nn, activation)())
        last_dim = dim
    layers.append(nn.Linear(last_dim, end_dim))

    return nn.Sequential(*layers)
