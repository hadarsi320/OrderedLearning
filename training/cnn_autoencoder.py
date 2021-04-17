import os

import numpy as np
from torch import nn, optim

import utils
from data import cifar10
from models.autoencoders import Autoencoder, ConvAutoencoder, NestedDropoutAutoencoder


def train_autoencoder(autoencoder: Autoencoder, dataloader, epochs, learning_rate, model_dir, **kwargs):
    nested_dropout = isinstance(autoencoder, NestedDropoutAutoencoder)
    plateau_limit = kwargs['plateau_limit'] if 'plateau_limit' in kwargs else 10
    batch_print = len(dataloader) // 5

    autoencoder.train()
    device = utils.get_device()
    autoencoder.to(device)

    optimizer = optim.Adam(params=autoencoder.parameters(), lr=learning_rate)
    if 'optimizer_state' in kwargs:
        optimizer.load_state_dict(kwargs.pop('optimizer_state'))
    loss_function = nn.MSELoss()

    losses = []
    best_loss = float('inf')
    plateau = 0
    for epoch in range(epochs):
        if nested_dropout and epoch > 0:
            print(f'\tEpoch {epoch + 1}/{epochs} '
                  f'({autoencoder.get_converged_unit()}/{autoencoder.get_dropout_dim()} converged units)')
        else:
            print(f'\tEpoch {epoch + 1}/{epochs}')

        batch_losses = []
        for i, (X, _) in enumerate(dataloader):
            optimizer.zero_grad()
            X = X.to(device)

            X_reconstruction = autoencoder(X)
            loss = loss_function(X, X_reconstruction)

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            if (i + 1) % batch_print == 0:
                print(f'Batch {i + 1} loss: {np.average(batch_losses[-batch_print:]):.3f}')

            if nested_dropout:
                autoencoder(X)
                if autoencoder.has_converged():
                    break

        epoch_loss = round(np.average(batch_losses), 3)
        losses.append(epoch_loss)
        print(f'\tTotal epoch loss {epoch_loss:.3f}\n')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            utils.save_model(autoencoder, optimizer, f'{model_dir}/model', losses=losses, epoch=epoch,
                             learning_rate=learning_rate, **kwargs)
            plateau = 0
        else:
            plateau += 1

        if plateau == plateau_limit or (nested_dropout is True and autoencoder.has_converged()):
            break

    return autoencoder, losses


def main():
    for filter_size in [2, 4, 8, 16, 32]:
        print(f'\t\tFilter size {filter_size}')
        epochs = 200
        learning_rate = 1e-3
        tol = 1e-3
        sequence_bound = 100

        model = NestedDropoutAutoencoder(ConvAutoencoder(filter_size=filter_size),
                                         tol=tol, sequence_bound=sequence_bound)
        dataloader = cifar10.get_dataloader(16)

        current_time = utils.current_time()
        model_name = f'cae-A-{type(model).__name__}_{current_time}'
        model_dir = f'{utils.save_dir}/{model_name}'
        os.mkdir(model_dir)

        train_autoencoder(model, dataloader, epochs, learning_rate, model_dir, filter_size=filter_size, tol=tol,
                          batch_norm=True, sequence_bound=sequence_bound, plateau_limit=None)

    # model_dict = torch.load('saves/cae-A-NestedDropoutAutoencoder_21-04-13--15-18-32/model.pt')
    # model = NestedDropoutAutoencoder(ConvAutoencoder(filter_size=model_dict['filter_size']))
    # model.load_state_dict(model_dict['model'])
    # dataloader = cifar10.get_dataloader(16)
    #
    # current_time = utils.current_time()
    # model_name = f'cae-A-{type(model).__name__}_{current_time}'
    # model_dir = f'{utils.save_dir}/{model_name}'
    # os.mkdir(model_dir)
    #
    # train_autoencoder(model, dataloader, 100, 1e-3, model_dir, optimizer_state=model_dict['optimizer'])
