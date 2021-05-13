import os
import time

import numpy as np
from torch import nn, optim

import training
import utils
from data import imagenette
from models.classifiers import Classifier


def train(classifier: Classifier, optimizer, dataloader, model_dir, epochs, **kwargs):
    nested_dropout = classifier.apply_nested_dropout
    plateau_limit = kwargs.get('plateau_limit', 10)
    loss_criterion = kwargs.get('loss_criterion', 'CrossEntropyLoss')
    loss_function = getattr(nn, loss_criterion)()
    batch_print = len(dataloader) // 5

    classifier.train()
    device = utils.get_device()
    classifier.to(device)

    losses = []
    best_loss = float('inf')
    plateau = 0
    train_time = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        line = f'\tEpoch {epoch + 1}/{epochs}'
        if nested_dropout:
            line += f' ({classifier.get_converged_unit()}/{classifier.get_dropout_dim()} converged units)'
        print(line)

        batch_losses = []
        for i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            X = X.to(device)
            prediction = classifier(X)
            loss = loss_function(prediction, X)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            if (i + 1) % batch_print == 0:
                batch_loss = utils.adaptable_round(np.average(batch_losses[-batch_print:]), 3)
                print(f'Batch {i + 1} loss: {batch_loss}')

            if nested_dropout:
                classifier(X)
                if classifier.has_converged():
                    break

        epoch_loss = utils.adaptable_round(np.average(batch_losses), 3)
        losses.append(epoch_loss)

        epoch_time = time.time() - epoch_start
        train_time += epoch_time

        print(f'\tTotal epoch loss {epoch_loss} \tEpoch time {utils.format_time(epoch_time)}\n')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            utils.save_model(classifier, optimizer, f'{model_dir}/model', losses=losses, best_loss=best_loss,
                             epoch=epoch, train_time=utils.format_time(train_time), **kwargs)
            plateau = 0
        else:
            plateau += 1

        if plateau == plateau_limit or (nested_dropout is True and classifier.has_converged()):
            break

    if nested_dropout is True and classifier.has_converged():
        end = 'nested dropout has converged'
    elif plateau == plateau_limit:
        end = 'has plateaued'
    else:
        end = f'reached max number of epochs ({epochs})'
    utils.update_save(f'{model_dir}/model', end=end)

    return losses


def main():
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


if __name__ == '__main__':
    main()
