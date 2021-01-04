from math import log2, ceil, floor

import torch
import torch.nn as nn

from torch import linalg


def create_sequential(start_dim: int, end_dim: int, activation: str = None, dropout=True, dropout_p=0.2):
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
        if dim in [start_dim, end_dim]:
            continue
        layers.append(nn.Linear(last_dim, dim))
        if activation is not None:
            layers.append(getattr(nn, activation)())
        if dropout:
            layers.append(nn.Dropout(p=dropout_p))
        last_dim = dim
    layers.append(nn.Linear(last_dim, end_dim))

    return nn.Sequential(*layers)


def nested_dropout(x: torch.Tensor, nested_dropout_dist, min_neuron: int):
    batch_size = x.shape[0]
    neurons = x.shape[1]

    dropout_sample = nested_dropout_dist.sample((batch_size,)).type(torch.long)
    dropout_sample[dropout_sample > (neurons - 1)] = neurons - 1

    mask = (torch.arange(neurons) <= (dropout_sample.unsqueeze(1) + min_neuron)).to(x.device)
    return mask * x


def code_variance(autoencoder, batch, batch_repr=None):
    if batch_repr is None:
        batch_repr = autoencoder.get_representation(batch)

    noise = torch.randn_like(batch)
    noisy_batch = batch + noise
    noisy_batch_repr = autoencoder.get_representation(noisy_batch)

    code_variance = torch.sum(linalg.norm(noisy_batch_repr, dim=1) / linalg.norm(noise, dim=1)) / len(batch)
    return code_variance
