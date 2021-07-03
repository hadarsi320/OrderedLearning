import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from clearml import Task
from clearml.logger import Logger
from ranger.ranger2020 import Ranger
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

import data
import model_visualizations
import utils
from models.autoencoders import ConvAutoencoder


def train(model: ConvAutoencoder, optimizer: optim.Optimizer, dataloader: DataLoader, epochs: int, loss_criterion: str,
          model_dir: str, plateau_limit: int, config: dict, lam: float = 1e-3, filter_prod_mode: str = None,
          lr_scheduler=None, logger: Logger = None, eval_dataloader: DataLoader = None):
    # TODO After making sure unoptimized nested dropout works, use only the fact that it is optimized for prints
    print(f'The model has {utils.get_num_parameters(model):,} parameters')
    loss_function = getattr(nn, loss_criterion)()

    device = utils.get_device()
    model.to(device)
    model.train()

    losses = []
    best_loss = float('inf')
    plateau = 0
    train_time = 0

    if logger is not None:
        report(logger, model, optimizer, lr_scheduler, epoch=0, train_loss=None)

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f'\tEpoch {epoch + 1}/{epochs}')

        batch_losses = []
        for i, (X, _) in enumerate(dataloader):
            optimizer.zero_grad()
            X = X.to(device)
            prediction = model(X)

            loss = loss_function(prediction, X)
            if filter_prod_mode is not None:
                filters, _ = model.get_weights(0)
                filters_product = utils.filter_correlation(filters, mode=filter_prod_mode)
                loss += lam * filters_product

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            if model.apply_nested_dropout and model.optimize_dropout and not model.has_converged():
                model(X)
                if model.has_converged():
                    break

        epoch_loss = utils.format_number(np.average(batch_losses))
        losses.append(epoch_loss)

        if eval_dataloader is None:
            eval_loss = None
        else:
            eval_loss = utils.format_number(utils.get_model_loss(model, eval_dataloader,
                                                                 loss_function=lambda x, y, res: F.mse_loss(res, x),
                                                                 device=device, show_progress=False))

        epoch_time = time.time() - epoch_start
        train_time += epoch_time

        print(f'Train loss {epoch_loss}')
        if eval_loss is not None:
            print(f'Eval loss {eval_loss}')
        if model.apply_nested_dropout:
            print(f'{model.get_converged_unit()}/{model.get_dropout_dim()} converged units')
        print(f'Epoch time {utils.format_time(epoch_time)}\n')

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
            report(logger, model, optimizer, lr_scheduler, epoch + 1, epoch_loss, eval_loss=eval_loss)

        if (plateau == plateau_limit) or model.has_converged():
            break

    if model.apply_nested_dropout is True and model.has_converged():
        end = 'Nested dropout has converged'
    elif plateau == plateau_limit:
        end = 'The model has plateaued'
    else:
        end = f'Reached max number of epochs ({epochs})'
    print(end)
    utils.update_save(f'{model_dir}/model', end=end)


def report(logger, model, optimizer, lr_scheduler, epoch, train_loss, eval_loss=None):
    if epoch > 0:
        logger.report_scalar('Model Loss', 'Train Loss', train_loss, iteration=epoch)
        if eval_loss is not None:
            logger.report_scalar('Model Loss', 'Eval Loss', eval_loss, iteration=epoch)

        if model.apply_nested_dropout:
            logger.report_scalar('Nested Dropout', 'Converged Unit',
                                 model.get_converged_unit(), iteration=epoch)
            logger.report_scalar('Nested Dropout', 'Dropout Dimension',
                                 model.get_dropout_dim(), iteration=epoch)

    model_visualizations.plot_filters(model, output_shape=(8, 8), show=False, normalize=True)
    logger.report_matplotlib_figure('Model Filters Normalized', 'Filters', figure=plt, iteration=epoch,
                                    report_image=True)
    plt.close()

    model_visualizations.plot_filters(model, output_shape=(8, 8), show=False, normalize=False)
    logger.report_matplotlib_figure('Model Filters Unnormalized', 'Filters', figure=plt, iteration=epoch,
                                    report_image=True)
    plt.close()
    if lr_scheduler is not None:
        logger.report_scalar('Learning Rate', 'Current', utils.get_learning_rate(optimizer),
                             iteration=epoch)
        logger.report_scalar('Learning Rate', 'Initial', optimizer.defaults['lr'],
                             iteration=epoch)

    weights, _ = model.get_weights(0)
    for mode in ['hadamund', 'frobenius']:
        logger.report_scalar('Filter Product', mode.title(), utils.filter_correlation(weights, mode),
                             iteration=epoch)


def train_cae(cfg: dict, task: Task = None):
    matplotlib.use('agg')

    data_module = getattr(data, cfg['data']['dataset'])
    if cfg['optim']['optimizer'] == 'Ranger':
        optimizer_model = Ranger
    else:
        optimizer_model = getattr(optim, cfg['optim']['optimizer'])

    train_dataloader = data_module.get_dataloader(**cfg['data']['kwargs'], train=True)
    eval_dataloader = data_module.get_dataloader(**cfg['data']['kwargs'], train=False)
    model = ConvAutoencoder(**cfg['model'], **cfg['nested_dropout'], **cfg['data']['kwargs'])
    optimizer = optimizer_model(params=model.parameters(), **cfg['optim']['optimizer_kwargs'])

    if 'lr_scheduler' in cfg['optim']:
        lr_scheduler_model = getattr(optim.lr_scheduler, cfg['optim']['lr_scheduler'])
        lr_scheduler = lr_scheduler_model(optimizer=optimizer, **cfg['optim']['lr_scheduler_kwargs'])
    else:
        lr_scheduler = None

    current_time = utils.get_current_time()
    model_name = f'{type(model).__name__}-{cfg["model"]["mode"]}_{current_time}'
    model_dir = f'{utils.save_dir}/{model_name}'

    if model.apply_nested_dropout:
        model_name += ' - Nested Dropout'

    if task is not None:
        task.set_name(model_name)
        logger = task.get_logger()
    else:
        logger = None

    train(model, optimizer, train_dataloader, model_dir=model_dir, config=cfg, lr_scheduler=lr_scheduler, logger=logger,
          **cfg['train'], eval_dataloader=eval_dataloader)

    if logger is not None:
        model.eval()
        model_visualizations.plot_images_by_channels(model, data_module,
                                                     normalized=cfg['data']['kwargs']['normalize'],
                                                     im_format='Y', show=False)
        logger.report_matplotlib_figure('Images by Num Channels', 'Images', figure=plt)
        plt.close()

        model_visualizations.plot_conv_autoencoder_reconstruction_error(model, train_dataloader, show=False)
        logger.report_matplotlib_figure('Reconstruction Error by Channels', 'Train Error', figure=plt)
        plt.close()

        model_visualizations.plot_conv_autoencoder_reconstruction_error(model, eval_dataloader, show=False)
        logger.report_matplotlib_figure('Reconstruction Error by Channels', 'Evaluation Error', figure=plt)
        plt.close()
