import os
from datetime import datetime, timedelta
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nueral_networks import autoencoders
from utils_package import data_utils
from utils_package import utils


def fit_autoencoder(autoencoder: autoencoders.Autoencoder, train_loader, learning_rate, epochs, model_name):
    # TODO add code invariance
    os.makedirs(f'checkpoints/{model_name}/')

    autoencoder.train(True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    autoencoder.to(device)

    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    batch_print = len(train_loader) // 5
    losses = []
    for epoch in range(epochs):
        print(f'\tEpoch {epoch + 1} ({autoencoder.conv_i}/{autoencoder.repr_dim} converged units)')
        batch_losses = []
        for i, (batch, _) in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = autoencoder(batch)
            loss = loss_function(batch, reconstruction)
            batch_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            autoencoder.check_convergence(batch)

            if (i + 1) % batch_print == 0:
                print(f'Batch {i + 1} loss: {np.average(batch_losses[-train_loader.batch_size:]):.3f}')

        epoch_loss = np.average(batch_losses)
        losses.append(epoch_loss)
        print(f'\tTotal epoch loss {epoch_loss:.3f}\n')

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # reps = utils.get_data_representation(autoencoder, train_loader, device)
            # reps_variance = torch.var(reps, dim=0)
            # plt.title(f'Representation variance- epoch {epoch + 1}')
            # plt.plot(reps_variance.cpu())
            # plt.savefig(f'plots/{model_name}_{epoch + 1}.png')
            # plt.show()
            torch.save(autoencoder, f'checkpoints/{model_name}/epoch_{epoch + 1}.pkl')

        if autoencoder.converged is True:
            print('\t\tThe autoencoder has converged')
            break
    print('Finished training')

    reps = utils.get_data_representation(autoencoder, train_loader, device)
    reps_variance = torch.var(reps, dim=0)
    plt.plot(reps_variance.to('cpu'))
    plt.title('Final representation variance')
    plt.show()
    torch.save(autoencoder, f'models/{model_name}.pkl')

    return losses


def main():
    epochs = 100
    learning_rate = 0.001
    batch_size = 1000
    rep_dim = 512
    activation = 'ReLU'

    model_name = f'nestedDropoutAutoencoder_deep_{rep_dim}_{activation}_' + datetime.now().strftime('%y_%m_%d_%H_%M_%S')

    train_dataset, train_loader = data_utils.load_cifar10(batch_size)
    autoencoder = autoencoders.Autoencoder(3072, rep_dim, apply_nested_dropout=True, activation=activation, deep=True,
                                           nested_dropout_p=0.03)
    print('The number of the model\'s parameters: {:,}'.format(sum(p.numel() for p in autoencoder.parameters())))
    losses = fit_autoencoder(autoencoder, train_loader, learning_rate, epochs, model_name)

    plt.plot(losses)
    plt.ylabel('Losses')
    plt.xlabel('Epochs')
    plt.savefig(f'plots/{model_name}/')

    # autoencoder = pickle.load(open('pickles/trained_autoencoder.pkl', 'rb'))


if __name__ == '__main__':
    start_time = time()
    main()
    print(f'Total run time: {timedelta(seconds=time() - start_time)}')
