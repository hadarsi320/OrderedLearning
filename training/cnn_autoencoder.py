import time

import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import utils
from data import imagenette
from models.autoencoders import ConvAutoencoder, NestedDropoutAutoencoder


def train(model: nn.Module, optimizer: optim.Optimizer, dataloader: DataLoader, epochs: int,
          loss_criterion: str, model_dir: str, plateau_limit: int, apply_nested_dropout: bool,
          **kwargs):
    print(f'The model has {utils.get_num_parameters(model):,} parameters')
    lr_scheduler = kwargs.pop('lr_scheduler', None)

    loss_function = getattr(nn, loss_criterion)()
    batch_print = len(dataloader) // 5

    model.train()
    device = utils.get_device()
    model.to(device)

    losses = []
    best_loss = float('inf')
    plateau = 0
    train_time = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        line = f'\tEpoch {epoch + 1}/{epochs}'
        if apply_nested_dropout and epoch > 0:
            line += f' ({model.get_converged_unit()}/{model.get_dropout_dim()} converged units)'
        print(line)

        batch_losses = []
        for i, (X, _) in enumerate(dataloader):
            optimizer.zero_grad()
            X = X.to(device)
            prediction = model(X)

            loss = loss_function(prediction, X)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            if (i + 1) % batch_print == 0:
                batch_loss = utils.format_number(np.average(batch_losses[-batch_print:]))
                print(f'Batch {i + 1} loss: {batch_loss}')

            if apply_nested_dropout:
                model(X)
                if model.has_converged():
                    break

        epoch_loss = utils.format_number(np.average(batch_losses))
        losses.append(epoch_loss)

        epoch_time = time.time() - epoch_start
        train_time += epoch_time

        print(f'\tEpoch loss {epoch_loss}')
        print(f'\tEpoch time {utils.format_time(epoch_time)}\n')

        model_save_kwargs = dict(**kwargs, epoch=epoch, train_time=utils.format_time(train_time), losses=losses)
        has_improved = False
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            has_improved = True
            model_save_kwargs.update(best_loss=best_loss)

        if lr_scheduler is not None:
            lr_scheduler.step(best_loss)

        if has_improved:
            utils.save_model(model, optimizer, f'{model_dir}/model', **model_save_kwargs)
            plateau = 0
        else:
            plateau += 1

        if (plateau == plateau_limit) or (apply_nested_dropout is True and model.has_converged()):
            break

    if apply_nested_dropout is True and model.has_converged():
        end = 'nested dropout has converged'
        print('Nested dropout has converged!')
    elif plateau == plateau_limit:
        end = 'has plateaued'
        print('The model has plateaued')
    else:
        end = f'reached max number of epochs ({epochs})'
        print('The maximum number of epochs has been reached')
    utils.update_save(f'{model_dir}/model', end=end)


def train_cae():
    # data options
    normalize_data = True
    dataset = 'imagenette'
    image_mode = 'Y'
    channels = 1 if image_mode == 'Y' else 3
    random_flip = False

    # optimization options
    loss_criterion = 'MSELoss'
    epochs = 50
    learning_rate = 1e-2
    batch_size = 16
    patience = 5
    plateau_limit = None

    # model options
    cae_mode = 'H'
    activation = 'ReLU'
    batch_norm = False

    # nested dropout options
    apply_nested_dropout = False
    dropout_depth = 1
    p = 0.1
    seq_bound = 2 ** 8
    tol = 1e-3

    dataloader = imagenette.get_dataloader(batch_size, normalize=normalize_data, image_mode=image_mode)
    model_kwargs = dict(mode=cae_mode, activation=activation, loss_criterion=loss_criterion, patience=patience,
                        learning_rate=learning_rate, batch_norm=batch_norm, dataset=dataset, image_mode=image_mode,
                        channels=channels, random_flip=random_flip, normalize_data=normalize_data,
                        plateau_limit=plateau_limit, apply_nested_dropout=apply_nested_dropout)
    model = ConvAutoencoder(**model_kwargs)

    if apply_nested_dropout:
        model_kwargs.update(dropout_depth=dropout_depth, p=p, sequence_bound=seq_bound, tol=tol)
        model = NestedDropoutAutoencoder(model, **model_kwargs)

    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, verbose=True)

    current_time = utils.get_current_time()
    model_name = f'cae-{cae_mode}-{type(model).__name__}_{current_time}'
    model_dir = f'{utils.save_dir}/{model_name}'

    train(model, optimizer, dataloader, lr_scheduler=lr_scheduler, model_dir=model_dir,
          epochs=epochs, **model_kwargs)
