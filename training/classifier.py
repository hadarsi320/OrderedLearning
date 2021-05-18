import os

from torch import optim

import training
import utils
from data import imagenette
from models.classifiers import Classifier


def train_classifier():
    # general options
    epochs = 50
    learning_rate = 1e-3
    normalize_data = True
    batch_size = 16
    loss_criterion = 'CrossEntropyLoss'
    plateau_limit = 5
    dataset = 'imagenette'
    image_mode = 'Y'
    channels = 1 if image_mode == 'Y' else 3

    # model options
    num_classes = 10
    activation = 'ReLU'
    batch_norm = True
    model_mode = 'A'

    # nested dropout options
    apply_nested_dropout = True
    dropout_depth = 1
    p = 0.1
    seq_bound = 2 ** 8
    tol = 1e-5

    dataloader = imagenette.get_dataloader(batch_size, normalize=normalize_data, image_mode=image_mode)
    model_kwargs = dict(num_classes=num_classes, mode=model_mode, activation=activation, loss_criterion=loss_criterion,
                        learning_rate=learning_rate, batch_norm=batch_norm, dataset=dataset, image_mode=image_mode,
                        channels=channels, normalize_data=normalize_data, plateau_limit=plateau_limit,
                        apply_nested_dropout=apply_nested_dropout)
    if apply_nested_dropout:
        model_kwargs.update(
            dict(dropout_depth=dropout_depth, p=p, sequence_bound=seq_bound, tol=tol))
    model = Classifier(**model_kwargs)
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    current_time = utils.current_time()
    model_name = f'classifier-{model_mode}-{type(model).__name__}_{current_time}'
    model_dir = f'{utils.save_dir}/{model_name}'
    os.mkdir(model_dir)

    print(f'The model has {utils.get_num_parameters(model)} parameters\n')
    return training.train(model, optimizer, dataloader, model_dir=model_dir, reconstruct=False, epochs=epochs,
                          nested_dropout=apply_nested_dropout, **model_kwargs)
