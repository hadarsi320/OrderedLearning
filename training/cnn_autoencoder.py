import os

from torch import optim

import training
import utils
from data import imagenette
from models.autoencoders import ConvAutoencoder, NestedDropoutAutoencoder


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
    activation = 'ReLU'
    batch_norm = False
    cae_mode = 'A'

    # nested dropout options
    dropout_depth = 1
    p = 0.1
    seq_bound = 2 ** 7
    tol = 1e-4

    dataloader = imagenette.get_dataloader(batch_size, normalize=normalize_data, image_mode=image_mode)
    model_kwargs = dict(mode=cae_mode, activation=activation, loss_criterion=loss_criterion,
                        learning_rate=learning_rate, batch_norm=batch_norm, dataset=dataset, image_mode=image_mode,
                        channels=channels, normalize_data=normalize_data, plateau_limit=plateau_limit)
    model = ConvAutoencoder(**model_kwargs)

    model_kwargs.update(dict(dropout_depth=dropout_depth, p=p, sequence_bound=seq_bound, tol=tol))
    model = NestedDropoutAutoencoder(model, **model_kwargs)

    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    if 'optimizer_state' in model_kwargs:
        optimizer.load_state_dict(model_kwargs.pop('optimizer_state'))

    current_time = utils.current_time()
    model_name = f'cae-{cae_mode}-{type(model).__name__}_{current_time}'
    model_dir = f'{utils.save_dir}/{model_name}'
    os.mkdir(model_dir)

    # train(model, optimizer, dataloader, model_dir, epochs, **model_kwargs)
    nested_dropout = isinstance(model, NestedDropoutAutoencoder)
    return training.train(model, optimizer, dataloader, model_dir=model_dir, reconstruct=True, epochs=epochs,
                          nested_dropout=nested_dropout, **model_kwargs)
