import os
from math import log2, ceil, floor

import numpy as np
import torch
import torch.nn as nn
from torch import linalg
from tqdm import tqdm
import yaml

import utils
from models import Classifier, ConvAutoencoder


def create_sequential(start_dim: int, end_dim: int, activation: str = None, dropout_p=0.2):
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
        if dropout_p is not None:
            layers.append(nn.Dropout(p=dropout_p))
        last_dim = dim
    layers.append(nn.Linear(last_dim, end_dim))

    return nn.Sequential(*layers)


@DeprecationWarning
def nested_dropout(x: torch.Tensor, nested_dropout_dist, min_neuron: int):
    batch_size = x.shape[0]
    neurons = x.shape[1]

    dropout_sample = nested_dropout_dist.sample((batch_size,)).type(torch.long)
    dropout_sample[dropout_sample > (neurons - 1)] = neurons - 1

    mask = (torch.arange(neurons) <= (dropout_sample.unsqueeze(1) + min_neuron)).to(x.device)
    return mask * x


def estimate_code_variance(autoencoder, batch, batch_repr=None):
    if batch_repr is None:
        batch_repr = autoencoder.encode(batch)

    noise = torch.randn_like(batch)
    noisy_batch = batch + noise
    noisy_batch_repr = autoencoder.encode(noisy_batch)

    code_variance = torch.sum(
        torch.pow(linalg.norm(noisy_batch_repr - batch_repr, dim=1) / linalg.norm(noise, dim=1), 2)) / len(batch)
    return code_variance


@torch.no_grad()
def get_model_loss(model, dataloader, loss_function, device=utils.get_device(), sample=None):
    model.eval()
    losses = []
    total = min(len(dataloader), sample) if sample is not None else len(dataloader)
    for i, (x, y) in tqdm(enumerate(dataloader), total=total):
        if i == total:
            break
        x, y = x.to(device), y.to(device)
        res = model(x)
        losses.append(loss_function(x, y, res).item())
    return np.average(losses)


@torch.no_grad()
def get_model_accuracy(model, dataloader, device=utils.get_device()):
    model.eval()
    correct = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        prediction = model.predict(x)
        correct.extend((prediction == y).numpy())
    return np.average(correct)


def save_model(model, optimizer, file_name, config):
    file_dir = os.path.dirname(file_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    if model.apply_nested_dropout:
        config['performance'].update(converged_unit=model.get_converged_unit(),
                                     has_converged=model.has_converged())

    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()),
               f'{file_name}.pt')
    utils.dump_yaml(config, f'{file_name}.yaml')


def update_save(file_name, **kwargs):
    if os.path.exists(f'{file_name}.yaml'):
        config = utils.load_yaml(f'{file_name}.yaml')
        config.update(kwargs)
        utils.dump_yaml(config, f'{file_name}.yaml')
    elif os.path.exists(f'{file_name}.txt'):
        with open(f'{file_name}.txt', 'a') as f:
            for key, value in kwargs.items():
                f.write(f'{key}: {value}\n')
    else:
        raise ValueError('No model file')


def fit_dim(tensor: torch.Tensor, target: torch.Tensor):
    while tensor.dim() < target.dim():
        tensor = tensor.unsqueeze(-1)
    return tensor


def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def get_data_representation(autoencoder, dataloader, device):
    return torch.cat([autoencoder.encode(batch.to(device)) for batch, _ in dataloader])


def load_model(model_save, device):
    dir_name = os.path.basename(model_save)
    if os.path.exists(f'{model_save}/model.yaml'):
        save_dict = utils.load_yaml(f'{model_save}/model.yaml')
        # TODO implement support for classifier
        if dir_name.startswith('cae'):
            model = ConvAutoencoder(**save_dict['model'], **save_dict['nested_dropout'], **save_dict['data']['kwargs'])
        else:
            raise NotImplementedError()

    # past compatibility
    else:
        save_dict = torch.load(f'{model_save}/model.pt', map_location=device)
        if dir_name.startswith('cae'):
            model = ConvAutoencoder(**save_dict)
        elif dir_name.startswith('classifier'):
            model = Classifier(**save_dict)
        else:
            raise NotImplementedError()

    try:
        states = torch.load(f'{model_save}/model.pt', map_location=device)
        model.load_state_dict(states['model'])
    except RuntimeError as e:
        print('Save dict mismatch\n')
        return None, None

    return save_dict, model
