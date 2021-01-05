import os

import torch

from nueral_networks.autoencoders import Autoencoder
from utils_package import utils, data_utils


def main():
    # autoencoder = torch.load(open(f'checkpoints/nestedDropoutAutoencoder_deep_512_ReLU_21_01_04_20_04_54/epoch_20.pkl',
    #                               'rb'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    autoencoder = Autoencoder(3072, 512, deep=True, activation='ReLU')
    autoencoder.eval()
    autoencoder.to(device)
    _, train_loader = data_utils.load_cifar10()
    utils.plot_repr_var(autoencoder, train_loader, device, show=True)


if __name__ == '__main__':
    main()
