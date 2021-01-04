import os
from datetime import datetime, timedelta
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn, linalg
from torch.distributions import Geometric

from nueral_networks.autoencoders import Autoencoder
from utils_package import data_utils, utils, nn_utils


def check_convergence(autoencoder, batch: torch.Tensor, old_repr: torch.Tensor, unit: int, succession: list, eps: float,
                      bound: int) -> bool:
    new_repr = autoencoder.get_representation(batch)
    if (linalg.norm(new_repr[:, unit] - old_repr[:, unit]) / batch.shape[0]) <= eps:
        succession[0] += 1
    else:
        succession[0] = 0

    if succession[0] == bound:
        succession[0] = 0
        return True
    return False


def fit_nested_dropout_autoencoder(autoencoder: Autoencoder, train_loader, learning_rate, epochs, model_name,
                                   nested_dropout_p=0.1, eps=1e-2, bound=20, corrupt_input=True, corrupt_p=0.1,
                                   demand_code_invariance=True):
    # TODO add code invariance
    os.makedirs(f'plots/{model_name}/')
    os.makedirs(f'checkpoints/{model_name}/')

    converged_unit = 0
    nested_dropout_dist = Geometric(probs=nested_dropout_p)
    converged = False
    succession = [0]

    corrupt_layer = nn.Dropout(p=corrupt_p)

    autoencoder.train(True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    autoencoder.to(device)

    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    batch_print = len(train_loader) // 5
    losses = []
    for epoch in range(epochs):
        print(f'\tEpoch {epoch + 1} ({converged_unit}/{autoencoder.repr_dim} converged units)')
        batch_losses = []
        for i, (batch, _) in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)

            # corrupt the input
            if corrupt_input:
                batch = corrupt_layer(batch)

            # run through the model and compute l2 loss
            representation = autoencoder.get_representation(batch)
            representation = nn_utils.nested_dropout(representation, nested_dropout_dist, converged_unit)
            reconstruction = autoencoder.get_reconstructions(representation)
            loss = loss_function(batch, reconstruction)
            loss.backward()

            # code invariance
            if demand_code_invariance:
                code_variance = nn_utils.code_variance(autoencoder, batch, representation)
                code_variance.backward()

            optimizer.step()

            batch_losses.append(loss.item())
            if (i + 1) % batch_print == 0:
                print(f'Batch {i + 1} loss: {np.average(batch_losses[-batch_print:]):.3f}')

            if check_convergence(autoencoder, batch, representation, converged_unit, succession, eps, bound):
                converged_unit += 1
                if converged_unit == autoencoder.repr_dim:
                    converged = True
                    break

        epoch_loss = np.average(batch_losses)
        losses.append(epoch_loss)
        print(f'\tTotal epoch loss {epoch_loss:.3f}\n')

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # utils.plot_repr_var(autoencoder, train_loader, device, show=False,
            #                     title=f'Representation variance- epoch {epoch + 1}',
            #                     savefig=f'plots/{model_name}/epoch_{epoch + 1}.png',)
            torch.save(autoencoder, f'checkpoints/{model_name}/epoch_{epoch + 1}.pkl')

        if converged is True:
            print('\t\tThe autoencoder has converged')
            break
    print('Finished training')

    # utils.plot_repr_var(autoencoder, train_loader, device,
    #                     savefig=f'models/{model_name}.pkl', title='Final representation variance')
    torch.save(autoencoder, f'models/{model_name}.pkl')

    plt.plot(losses)
    plt.ylabel('Losses')
    plt.xlabel('Epochs')
    plt.savefig(f'plots/{model_name}/losses')


def main():
    epochs = 100
    learning_rate = 0.001
    batch_size = 1000
    rep_dim = 512
    activation = 'ReLU'

    train_dataset, train_loader = data_utils.load_cifar10(batch_size)

    model_name = f'nestedDropoutAutoencoder_shallow_100_corrupt'
    autoencoder = Autoencoder(3072, 100, deep=False)
    # model_name = f'nestedDropoutAutoencoder_deep_{rep_dim}_{activation}_' + datetime.now().strftime(
    #     '%y_%m_%d_%H_%M_%S')
    # autoencoder = Autoencoder(3072, rep_dim, deep=True, activation=activation)

    print('The number of the model\'s parameters: {:,}'.format(sum(p.numel() for p in autoencoder.parameters())))
    fit_nested_dropout_autoencoder(autoencoder, train_loader, learning_rate, epochs, model_name, corrupt_input=True,
                                   demand_code_invariance=False)

    # autoencoder = pickle.load(open('pickles/trained_autoencoder.pkl', 'rb'))


if __name__ == '__main__':
    start_time = time()
    main()
    print(f'Total run time: {timedelta(seconds=time() - start_time)}')
