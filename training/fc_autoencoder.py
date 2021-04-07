import itertools
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn, linalg
from torch.distributions import Geometric
from tqdm import tqdm

import utils
from models.autoencoders import FCAutoencoder
from data import cifar10


def check_unit_convergence(autoencoder, batch: torch.Tensor, old_repr: torch.Tensor, unit: int, succession: list,
                           eps: float, bound: int) -> bool:
    new_repr = autoencoder.encode(batch)

    difference = linalg.norm((new_repr - old_repr)[:, :unit + 1]) / (len(batch) * (unit + 1))
    if difference <= eps:
        succession[0] += 1
    else:
        succession[0] = 0

    if succession[0] == bound:
        succession[0] = 0
        return True
    return False


def fit_vanilla_autoencoder(autoencoder: FCAutoencoder, dataloader, learning_rate, epochs, model_name,
                            corrupt_input=True, corrupt_p=0.1, demand_code_invariance=True, epoch_print=5,
                            save_plots=True, save_models=True, show_plots=False):
    if save_plots:
        os.makedirs(f'plots/{model_name}/')
    if save_models:
        os.makedirs(f'checkpoints/{model_name}/')

    corrupt_layer = nn.Dropout(p=corrupt_p)

    autoencoder.train(True)
    device = utils.get_device()
    autoencoder.to(device)

    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    batch_print = len(dataloader) // 5
    losses = []
    for epoch in tqdm(range(epochs)):
        print(f'\tEpoch {epoch + 1}/{epochs}')
        batch_losses = []
        for i, (batch, _) in enumerate(dataloader):
            optimizer.zero_grad()
            batch = batch.to(device)

            # corrupt the input
            if corrupt_input:
                batch = corrupt_layer(batch)

            # run through the model and compute l2 loss
            representation = autoencoder.encode(batch)
            reconstruction = autoencoder.decode(representation)
            loss = loss_function(batch, reconstruction)

            # code invariance
            if demand_code_invariance:
                code_variance = utils.estimate_code_variance(autoencoder, batch, representation)
                loss += code_variance

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            if (i + 1) % batch_print == 0:
                print(f'Batch {i + 1} loss: {np.average(batch_losses[-batch_print:]):.3f}')

        epoch_loss = np.average(batch_losses)
        losses.append(epoch_loss)
        print(f'\tTotal epoch loss {epoch_loss:.3f}\n')

        if (epoch + 1) % epoch_print == 0 or epoch == 0:
            kwargs = {}
            if save_plots:
                kwargs['savefig'] = f'plots/{model_name}/epoch_{epoch + 1}.png'
            if save_plots or show_plots:
                utils.plot_repr_var(autoencoder, dataloader, device, show=show_plots,
                                    title=f'Representation variance- epoch {epoch + 1}',
                                    **kwargs)
            if save_models:
                torch.save(autoencoder, f'checkpoints/{model_name}/epoch_{epoch + 1}.pt')
    print('Finished training')

    kwargs = {}
    if save_plots:
        kwargs['savefig'] = f'plots/{model_name}/final.png'
    if save_plots or show_plots:
        utils.plot_repr_var(autoencoder, dataloader, device,
                            title='Final representation variance', show=show_plots, **kwargs)

        plt.clf()
        plt.plot(losses)
        plt.ylabel('Losses')
        plt.xlabel('Epochs')
        if save_plots:
            plt.savefig(f'plots/{model_name}/losses')
        if show_plots:
            plt.show()

    if save_models:
        torch.save(autoencoder, f'models/{model_name}.pt')

    output = {'autoencoder': autoencoder, 'losses': losses, 'optimizer_state': optimizer.state_dict()}
    return output


def fit_nested_dropout_autoencoder(autoencoder: FCAutoencoder, dataloader, learning_rate, epochs, model_name,
                                   nested_dropout_p=0.1, eps=1e-2, bound=20, corrupt_input=True, corrupt_p=0.1,
                                   demand_code_invariance=True, epoch_print=5, save_plots=True, save_models=True,
                                   show_plots=False):
    if save_plots:
        os.makedirs(f'plots/{model_name}/')
    if save_models:
        os.makedirs(f'checkpoints/{model_name}/')

    converged_unit = 0
    nested_dropout_dist = Geometric(probs=nested_dropout_p)
    converged = False
    succession = [0]

    corrupt_layer = nn.Dropout(p=corrupt_p)

    autoencoder.train(True)
    device = utils.get_device()
    autoencoder.to(device)

    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    batch_print = len(dataloader) // 5
    losses = []
    for epoch in tqdm(range(epochs)):
        print(f'\tEpoch {epoch + 1}/{epochs} ({converged_unit}/{autoencoder.repr_dim} converged units)')
        batch_losses = []
        for i, (batch, _) in enumerate(dataloader):
            optimizer.zero_grad()
            batch = batch.to(device)

            # corrupt the input
            if corrupt_input:
                batch = corrupt_layer(batch)

            # run through the model and compute l2 loss
            representation = autoencoder.encode(batch)
            representation = utils.nested_dropout(representation, nested_dropout_dist, converged_unit)
            reconstruction = autoencoder.decode(representation)
            loss = loss_function(batch, reconstruction)

            # code invariance
            if demand_code_invariance:
                code_variance = utils.estimate_code_variance(autoencoder, batch, representation)
                loss += code_variance

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            if (i + 1) % batch_print == 0:
                print(f'Batch {i + 1} loss: {np.average(batch_losses[-batch_print:]):.3f}')

            if check_unit_convergence(autoencoder, batch, representation, converged_unit, succession, eps, bound):
                converged_unit += 1
                if converged_unit == autoencoder.repr_dim:
                    converged = True
                    break

        epoch_loss = np.average(batch_losses)
        losses.append(epoch_loss)
        print(f'\tTotal epoch loss {epoch_loss:.3f}\n')

        if (epoch + 1) % epoch_print == 0 or epoch == 0:
            kwargs = {}
            if save_plots:
                kwargs['savefig'] = f'plots/{model_name}/epoch_{epoch + 1}.png'
            if save_plots or show_plots:
                utils.plot_repr_var(autoencoder, dataloader, device, show=show_plots,
                                    title=f'Representation variance- epoch {epoch + 1}',
                                    **kwargs)
            if save_models:
                # TODO implement saving a model when its accuracy is higher than its priors
                torch.save(autoencoder, f'checkpoints/{model_name}/epoch_{epoch + 1}.pt')

        if converged is True:
            print('\t\tThe autoencoder has converged')
            break
    print('Finished training')

    kwargs = {}
    if save_plots:
        kwargs['savefig'] = f'plots/{model_name}/final.png'
    if save_plots or show_plots:
        utils.plot_repr_var(autoencoder, dataloader, device,
                            title='Final representation variance', show=show_plots, **kwargs)

        plt.clf()
        plt.plot(losses)
        plt.ylabel('Losses')
        plt.xlabel('Epochs')
        if save_plots:
            plt.savefig(f'plots/{model_name}/losses')
        if show_plots:
            plt.show()

    if save_models:
        torch.save(autoencoder, f'models/{model_name}.pt')

    output = {'converged_unit': converged_unit, 'autoencoder': autoencoder, 'losses': losses,
              'optimizer_state': optimizer.state_dict()}

    return output


def test_params(batch_size, learning_rate, eps, bound, deep, repr_dim, epochs, nested_dropout_p, activation='ReLU',
                epoch_print=10, nested_dropout=True):
    model_params = locals()
    deep_str = 'deep' if deep else 'shallow'

    dataloader = cifar10.get_dataloader(batch_size)
    autoencoder = FCAutoencoder(3072, repr_dim, deep=deep, activation=activation)

    if nested_dropout:
        model_name = f'nestedDropoutAutoencoder_{deep_str}_{utils.current_time()}'
        print('Training nested dropout autoencoder')
        output = fit_nested_dropout_autoencoder(autoencoder, dataloader, learning_rate,
                                                epochs, model_name,
                                                nested_dropout_p=nested_dropout_p,
                                                bound=bound, epoch_print=epoch_print,
                                                save_models=True, save_plots=False, eps=eps)
    else:
        model_name = f'vanillaAutoencoder_{deep_str}_{utils.current_time()}'
        print('Training vanilla autoencoder')
        output = fit_vanilla_autoencoder(autoencoder, dataloader, learning_rate, epochs, model_name)
    output['parameters'] = model_params
    output['nested_dropout'] = nested_dropout
    torch.save(obj=output, f=f'models/{model_name}_dict.pt')


def main():
    # batch_size = [1000]
    # learning_rate = [1e-3]
    # eps = [1e-3]
    # bound = [10, 20]
    # deep = [False]
    # repr_dim = [1024]
    # epochs = [500]
    # p = [0.1]
    #
    # parameters = itertools.product(batch_size, learning_rate, eps, bound, deep, repr_dim, epochs, p)
    # for params in parameters:
    #     test_params(*params, nested_dropout=False)

    test_params(1000, 1e-3, None, None, False, 1024, 500, None, nested_dropout=False)
