from time import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils_package import data_utils
from nueral_networks import autoencoders
from utils_package import utils


def flatten(x: torch.Tensor):
    return x.view(-1)


def plot_representation_variance(autoencoder, dataloader):
    reps = utils.get_data_representation(autoencoder, dataloader)

    reps_variance = torch.var(reps, dim=0)
    plt.plot(reps_variance)
    plt.show()


def fit_autoencoder(autoencoder, train_loader, learning_rate, epochs, model_name):
    autoencoder.train(True)
    autoencoder = autoencoder

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    batch_print = len(train_loader) // 5
    losses = []
    for epoch in range(epochs):
        print(f'\tEpoch {epoch + 1} ({autoencoder.conv_i}/100 converged units)')
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
            torch.save(autoencoder, f'checkpoints/{model_name}_epoch_{epoch+1}.pkl')
            # reps = utils.get_data_representation(autoencoder, train_dataset)
            # reps_variance = torch.var(reps, dim=0)
            # plt.title(f'Representation variance- epoch {epoch + 1}')
            # plt.plot(reps_variance.cpu())
            # plt.savefig(f'plots/{model_name}_{epoch + 1}.png')
            # plt.show()

        if autoencoder.converged is True:
            print('\t\tThe autoencoder converged prematurely')
            break
        # if epoch_loss >= last_loss:
        #     break
        # last_loss = epoch_loss
    print('Finished training')
    return losses


def main():
    epochs = 1
    learning_rate = 0.001
    batch_size = 1000

    model_name = 'nested_dropout_autoencoder_' + datetime.now().strftime('%H_%M_%S')

    train_dataset, train_loader = data_utils.load_cifar10(batch_size)
    autoencoder = autoencoders.Autoencoder(3072, 100, apply_nested_dropout=True)
    losses = fit_autoencoder(autoencoder, train_loader, learning_rate, epochs, model_name)
    torch.save(autoencoder, f'models/{model_name}.pkl')

    plt.plot(losses)
    plt.ylabel('Losses')
    plt.xlabel('Epochs')
    plt.show()

    # autoencoder = pickle.load(open('pickles/trained_autoencoder.pkl', 'rb'))

    reps = utils.get_data_representation(autoencoder, train_loader)
    reps_variance = torch.var(reps, dim=0)
    plt.plot(reps_variance.to('cpu'))
    plt.title('Final representation variance')
    plt.show()


if __name__ == '__main__':
    start_time = time()
    main()
    print(f'Total run time: {time()-start_time}')
