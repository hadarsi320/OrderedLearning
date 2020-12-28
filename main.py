import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import data
import neural_networks
import utils


def flatten(x: torch.Tensor):
    return x.view(-1)


def plot_representation_variance(autoencoder, trainset):
    reps = utils.get_data_representation(autoencoder, trainset)

    reps_variance = torch.var(reps, dim=0)
    plt.plot(reps_variance)
    plt.show()


def fit_autoencoder(autoencoder, trainset, train_loader, learning_rate, epochs, batch_size):
    autoencoder.train(True)
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    batch_print = len(train_loader) // 5
    last_loss = float('inf')
    for epoch in range(epochs):
        print(f'\tEpoch {epoch + 1} ({autoencoder.conv_i}/100 converged units)')
        batch_losses = []
        for i, (batch, _) in enumerate(train_loader):
            optimizer.zero_grad()
            reconstruction = autoencoder(batch)
            loss = loss_function(batch, reconstruction)
            batch_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            autoencoder.check_convergence(batch)

            if (i + 1) % batch_print == 0:
                print(f'Batch {i + 1} loss: {np.average(batch_losses[-batch_size:]):.3f}')

        epoch_loss = np.average(batch_losses)
        print(f'\tTotal epoch loss {epoch_loss:.3f}\n')

        if (epoch + 1) % 5 == 0 or epoch == 0:
            reps = utils.get_data_representation(autoencoder, trainset)
            reps_variance = torch.var(reps, dim=0)
            plt.title(f'Representation variance- epoch {epoch + 1}')
            plt.plot(reps_variance)
            plt.show()
            plt.savefig(f'plots/nested_dropout_autoencoder_epoch_{epoch+1}')

        if autoencoder.converged is True:
            print('\t\tThe autoencoder converged prematurely')
            break
        # if epoch_loss >= last_loss:
        #     break
        # last_loss = epoch_loss
    print('Finished training')


def main():
    epochs = 100
    learning_rate = 0.001
    batch_size = 1000

    trainset, train_loader = data.load_cifar10(batch_size)
    autoencoder = neural_networks.Autoencoder(3072, 100, apply_nested_dropout=True)
    fit_autoencoder(autoencoder, trainset, train_loader, learning_rate, epochs, batch_size)
    torch.save(autoencoder, 'pickles/nested_dropout_autoencoder.pkl')

    # autoencoder = pickle.load(open('pickles/trained_autoencoder.pkl', 'rb'))

    with torch.no_grad():
        reps = torch.stack(
            [autoencoder.get_repr(x) for x, _ in trainset]).squeeze()

    reps_variance = torch.var(reps, dim=0)
    plt.plot(reps_variance)
    plt.title('Final representation variance')
    plt.show()


if __name__ == '__main__':
    main()
