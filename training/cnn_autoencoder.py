import os

import numpy as np
from torch import nn, optim

import utils
from data import cifar10
from models.autoencoders import Autoencoder, ConvAutoencoder


def train_autoencoder(autoencoder: Autoencoder, dataloader, epochs, learning_rate, model_dir, **kwargs):
    batch_print = len(dataloader) // 5

    autoencoder.train()
    device = utils.get_device()
    autoencoder.to(device)

    optimizer = optim.Adam(params=autoencoder.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    losses = []
    best_loss = float('inf')
    plateau = 0
    for epoch in range(epochs):
        print(f'\tEpoch {epoch + 1}/{epochs}')

        batch_losses = []
        for i, (batch, _) in enumerate(dataloader):
            optimizer.zero_grad()
            batch = batch.to(device)

            reconstruction = autoencoder(batch)
            loss = loss_function(batch, reconstruction)

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            if (i + 1) % batch_print == 0:
                print(f'Batch {i + 1} loss: {np.average(batch_losses[-batch_print:]):.3f}')

        epoch_loss = round(np.average(batch_losses), 3)
        losses.append(epoch_loss)
        print(f'\tTotal epoch loss {epoch_loss:.3f}\n')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            utils.save_model(autoencoder, optimizer, f'{model_dir}/model', losses=losses, epoch=epoch, **kwargs)
            plateau = 0
        else:
            plateau += 1

        if plateau == 10:
            break

    return autoencoder, losses


def main():
    for filter_size in [2, 4, 8, 16, 32]:
        current_time = utils.current_time()
        model_name = f'simple-cae_{current_time}'
        model_dir = f'{utils.save_dir}/{model_name}'
        os.mkdir(model_dir)

        epochs = 100
        learning_rate = 1e-3

        model = ConvAutoencoder(filter_size=filter_size)
        dataloader = cifar10.get_dataloader(64)

        train_autoencoder(model, dataloader, epochs, learning_rate, model_dir, filter_size=filter_size)


if __name__ == '__main__':
    main()
