import os

import numpy as np
from torch import nn, optim

import utils
from data import imagenette
from models.autoencoders import Autoencoder, ConvAutoencoder, NestedDropoutAutoencoder


def train_autoencoder(autoencoder: Autoencoder, dataloader, model_dir, epochs, learning_rate, **kwargs):
    nested_dropout = isinstance(autoencoder, NestedDropoutAutoencoder)
    plateau_limit = kwargs.get('plateau_limit', 10)
    loss_criterion = kwargs.get('loss_criterion', 'MSELoss')
    batch_print = len(dataloader) // 5

    autoencoder.train()
    device = utils.get_device()
    autoencoder.to(device)

    optimizer = optim.Adam(params=autoencoder.parameters(), lr=learning_rate)
    if 'optimizer_state' in kwargs:
        optimizer.load_state_dict(kwargs.pop('optimizer_state'))
    loss_function = getattr(nn, loss_criterion)()

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
            loss = loss_function(X_reconstruction, X)

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
            utils.save_model(autoencoder, optimizer, f'{model_dir}/model', losses=losses, best_loss=best_loss,
                             epoch=epoch, learning_rate=learning_rate, **kwargs)
            plateau = 0
        else:
            plateau += 1

        if plateau == plateau_limit or (nested_dropout is True and autoencoder.has_converged()):
            break

    if nested_dropout is True and autoencoder.has_converged():
        end = 'nested dropout has converged'
    elif plateau == plateau_limit:
        end = 'has plateaued'
    else:
        end = f'reached max number of epochs ({epochs})'
    utils.update_save(f'{model_dir}/model', end=end)

    return autoencoder, losses


def main():
    # general options
    epochs = 50
    learning_rate = 1e-3
    normalize_data = True
    batch_size = 16
    loss_criterion = 'MSELoss'
    plateau_limit = 5
    dataset = 'imagenette'
    image_mode = 'Y'
    channels = 1 if image_mode == 'Y' else 3

    # model options
    batch_norm = True
    cae_mode = 'E'

    # nested dropout options
    tol = 1e-4
    seq_bound = 2 ** 7
    p = 0.1
    dropout_depth = 1

    dataloader = imagenette.get_dataloader(batch_size, normalize=normalize_data, image_mode=image_mode)
    model_kwargs = dict(mode=cae_mode, loss_criterion=loss_criterion, learning_rate=learning_rate,
                        batch_norm=batch_norm, dataset=dataset, image_mode=image_mode, channels=channels,
                        normalize_data=normalize_data, plateau_limit=plateau_limit)
    model = ConvAutoencoder(**model_kwargs)

    model_kwargs.update(dict(dropout_depth=dropout_depth, p=p, sequence_bound=seq_bound, tol=tol))
    model = NestedDropoutAutoencoder(model, **model_kwargs)

    current_time = utils.current_time()
    model_name = f'cae-{cae_mode}-{type(model).__name__}_{current_time}'
    model_dir = f'{utils.save_dir}/{model_name}'
    os.mkdir(model_dir)

    train_autoencoder(model, dataloader, model_dir, epochs, **model_kwargs)
