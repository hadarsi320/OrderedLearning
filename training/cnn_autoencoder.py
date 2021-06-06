import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from clearml import Task
from clearml.logger import Logger
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import data
import model_visualizations
import utils
from models.autoencoders import ConvAutoencoder


def train(model: ConvAutoencoder, optimizer: optim.Optimizer, dataloader: DataLoader, epochs: int, loss_criterion: str,
          model_dir: str, plateau_limit: int, config: dict, lr_scheduler=None, logger: Logger = None):
    # TODO After making sure unoptimized nested dropout works, use only the fact that it is optimized for prints
    print(f'The model has {utils.get_num_parameters(model):,} parameters')
    loss_function = getattr(nn, loss_criterion)()
    batch_print = len(dataloader) // 5

    device = utils.get_device()
    model.to(device)
    model.train()

    losses = []
    best_loss = float('inf')
    plateau = 0
    train_time = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        line = f'\tEpoch {epoch + 1}/{epochs}'
        if model.apply_nested_dropout and epoch > 0:
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

            if model.apply_nested_dropout and model.optimize_dropout and not model.has_converged():
                model(X)
                # if model.has_converged():
                #     break

        epoch_loss = utils.format_number(np.average(batch_losses))
        losses.append(epoch_loss)

        epoch_time = time.time() - epoch_start
        train_time += epoch_time

        print(f'\tEpoch loss {epoch_loss}')
        print(f'\tEpoch time {utils.format_time(epoch_time)}\n')

        model_save_dict = config.copy()
        model_save_dict['performance'] = dict(epoch=epoch, train_time=utils.format_time(train_time), losses=losses)
        has_improved = False
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            has_improved = True
            model_save_dict['performance']['best_loss'] = best_loss

        if model.apply_nested_dropout:
            model_save_dict['performance']['converged_unit'] = model.get_converged_unit()
            model_save_dict['performance']['dropout_dim'] = model.get_dropout_dim()

        # if lr_scheduler is not None and (model.apply_nested_dropout and model.has_converged()):
        if lr_scheduler is not None:
            if type(lr_scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                lr_scheduler.step(epoch_loss)
            else:
                lr_scheduler.step()

        utils.save_model(model, optimizer, f'{model_dir}/model', model_save_dict)
        if has_improved:
            utils.save_model(model, optimizer, f'{model_dir}/best_model', model_save_dict)
            plateau = 0
        else:
            plateau += 1

        if logger is not None:
            iteration = epoch + 1
            logger.report_scalar('Model Loss', 'Train Loss', epoch_loss, iteration=iteration)
            model_visualizations.plot_filters(model, output_shape=(8, 8), show=False)
            logger.report_matplotlib_figure('Model Filters', 'Filters', figure=plt, iteration=iteration,
                                            report_image=True)
            plt.close()
            if lr_scheduler is not None:
                logger.report_scalar('Learning Rate', 'Current', utils.get_learning_rate(optimizer),
                                     iteration=iteration)
                logger.report_scalar('Learning Rate', 'Initial', optimizer.defaults['lr'],
                                     iteration=iteration)
            if model.apply_nested_dropout:
                logger.report_scalar('Nested Dropout', 'Converged Unit',
                                     model.get_converged_unit(), iteration=iteration)
                logger.report_scalar('Nested Dropout', 'Dropout Dimension',
                                     model.get_dropout_dim(), iteration=iteration)

        # if (plateau == plateau_limit) or (apply_nested_dropout is True and model.has_converged()):
        #     break

    if model.apply_nested_dropout is True and model.has_converged():
        end = 'Nested dropout has converged'
    elif plateau == plateau_limit:
        end = 'The model has plateaued'
    else:
        end = f'Reached max number of epochs ({epochs})'
    print(end)
    utils.update_save(f'{model_dir}/model', end=end)


def train_cae(cfg):
    matplotlib.use('agg')

    task = Task.init(project_name="Ordered Learning")
    task.connect_configuration(cfg, name='Model Configuration')

    data_module = getattr(data, cfg['data']['dataset'])
    optimizer_model = getattr(optim, cfg['optim']['optimizer'])

    dataloader = data_module.get_dataloader(**cfg['data']['kwargs'])
    model = ConvAutoencoder(**cfg['model'], **cfg['nested_dropout'], **cfg['data']['kwargs'])
    optimizer = optimizer_model(params=model.parameters(), **cfg['optim']['optimizer_kwargs'])

    if 'lr_scheduler' in cfg['optim']:
        lr_scheduler_model = getattr(optim.lr_scheduler, cfg['optim']['lr_scheduler'])
        lr_scheduler = lr_scheduler_model(optimizer=optimizer, **cfg['optim']['lr_scheduler_kwargs'])
    else:
        lr_scheduler = None

    current_time = utils.get_current_time()
    model_name = f'cae-{cfg["model"]["mode"]}-{type(model).__name__}_{current_time}'
    model_dir = f'{utils.save_dir}/{model_name}'

    if model.apply_nested_dropout:
        model_name += ' - Nested Dropout'
    task.set_name(model_name)
    logger = task.get_logger()

    train(model, optimizer, dataloader, model_dir=model_dir, config=cfg, lr_scheduler=lr_scheduler, logger=logger,
          **cfg['train'])
