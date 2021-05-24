import time

import numpy as np
from torch import nn

import utils
from torch import optim
from torch.utils.data import DataLoader

__all__ = ['train']


def train(model: nn.Module, optimizer: optim.Optimizer, dataloader: DataLoader, epochs: int,
          loss_criterion: str, model_dir: str, plateau_limit: int, nested_dropout: bool, reconstruct: bool, **kwargs):
    print(f'The model has {utils.get_num_parameters(model)} parameters')
    testloader = kwargs.pop('testloader', None)

    loss_function = getattr(nn, loss_criterion)()
    batch_print = len(dataloader) // 5

    model.train()
    device = utils.get_device()
    model.to(device)  # TODO check if this actually does anything

    losses = []
    accuracies = []
    best_loss = float('inf')
    best_accuracy = 0
    plateau = 0
    train_time = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        line = f'\tEpoch {epoch + 1}/{epochs}'
        if nested_dropout and epoch > 0:
            line += f' ({model.get_converged_unit()}/{model.get_dropout_dim()} converged units)'
        print(line)

        batch_losses = []
        for i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            prediction = model(X)

            if reconstruct:
                loss = loss_function(prediction, X)
            else:
                loss = loss_function(prediction, y)

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            if (i + 1) % batch_print == 0:
                batch_loss = utils.adaptable_round(np.average(batch_losses[-batch_print:]), 3)
                print(f'Batch {i + 1} loss: {batch_loss}')

            if nested_dropout:
                model(X)
                if model.has_converged():
                    break

        epoch_loss = utils.adaptable_round(np.average(batch_losses), 3)
        losses.append(epoch_loss)

        epoch_time = time.time() - epoch_start
        train_time += epoch_time

        print(f'\tTotal epoch loss {epoch_loss}')

        model_save_kwargs = dict(**kwargs, epoch=epoch, train_time=utils.format_time(train_time), losses=losses)
        has_improved = False
        if testloader is not None:
            model.eval()
            eval_accuracy = round(utils.get_model_accuracy(model, testloader, device), 3)
            model.train()
            accuracies.append(eval_accuracy)
            print(f'\tEvaluation accuracy {eval_accuracy}')

            if eval_accuracy > best_accuracy:
                best_accuracy = eval_accuracy
                has_improved = True
                model_save_kwargs.update(accuracies=accuracies, best_accuracy=best_accuracy)

        elif epoch_loss < best_loss:
            best_loss = epoch_loss
            has_improved = True
            model_save_kwargs.update(best_loss=best_loss)

        print(f'\tEpoch time {utils.format_time(epoch_time)}\n')
        if has_improved:
            utils.save_model(model, optimizer, f'{model_dir}/model', **model_save_kwargs)
            plateau = 0
        else:
            plateau += 1

        if (plateau == plateau_limit) or (nested_dropout is True and model.has_converged()):
            break

    if nested_dropout is True and model.has_converged():
        end = 'nested dropout has converged'
        print('Nested dropout has converged!')
    elif plateau == plateau_limit:
        end = 'has plateaued'
        print('The model has plateaued...')
    else:
        end = f'reached max number of epochs ({epochs})'
        print('The maximum number of epochs has been reached...')
    utils.update_save(f'{model_dir}/model', end=end)

    return losses
