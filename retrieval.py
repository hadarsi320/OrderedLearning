import pickle

import torch
from torchvision.transforms import ToPILImage

from data_structures.binary_tree import BinaryTree
from nueral_networks.autoencoders import Autoencoder
from utils_package import data_utils, utils

import matplotlib.pyplot as plt

import random


def linear_scan(item, data, repr):
    for i, _item in enumerate(repr):
        if item == _item:
            return data[i]
    raise ValueError('item not in repr')


def main():
    model_pickle = f'models/nestedDropoutAutoencoder_deep_ReLU_21-01-07__01-18-13.pkl'
    _, dataloader = data_utils.get_cifar10_dataloader(1000)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    autoencoder: Autoencoder = torch.load(open(model_pickle, 'rb'), map_location=device)

    data_repr = torch.stack([autoencoder.get_representation(batch.to(device)) for batch, _ in dataloader])

    # random.seed(420)
    # for sample, _ in dataset:
    #     if random.random() <= 0.9:
    #         continue
    #     plt.imshow(utils.restore_image(sample.view(3, 32, 32), mean, std))
    #     plt.show()
    #
    #     with torch.no_grad():
    #         reconstructed = autoencoder(sample).to('cpu')
    #         plt.imshow(utils.restore_image(reconstructed.view(3, 32, 32), mean, std))
    #         plt.show()
    #     break


if __name__ == '__main__':
    main()
