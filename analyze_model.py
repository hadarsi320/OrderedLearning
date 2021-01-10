import os

import torch

from nueral_networks.autoencoders import Autoencoder
from utils_package import utils, data_utils


def main():
    autoencoder: Autoencoder = \
        torch.load(open(f'models/nestedDropoutAutoencoder_deep_deep_ReLU_21-01-07__01-18-13.pkl', 'rb'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    autoencoder.eval()
    autoencoder.to(device)
    _, train_loader = data_utils.load_cifar10()
    utils.plot_repr_var(autoencoder, train_loader, device, show=True)


if __name__ == '__main__':
    main()
