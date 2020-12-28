import torch
from torchvision.transforms import ToPILImage

from data_structures.binary_tree import BinaryTree
from utils_package import data_utils, utils

import matplotlib.pyplot as plt

import random


def main():
    mean = [0.49139968, 0.48215841, 0.44653091]
    std = [0.24703223, 0.24348513, 0.26158784]
    dataset, dataloader = data_utils.load_cifar10(100)
    autoencoder = torch.load(open('models/nested_dropout_autoencoder_14_00_30.pkl', 'rb'))

    # data_representation = utils.get_data_representation(autoencoder, dataloader)
    random.seed(42)
    for sample, _ in dataset:
        if random.random() <= 0.9:
            continue
        plt.imshow(utils.restore_image(sample.view(3, 32, 32), mean, std))
        plt.show()

        with torch.no_grad():
            reconstructed = autoencoder(sample).to('cpu')
            plt.imshow(utils.restore_image(reconstructed.view(3, 32, 32), mean, std))
            plt.show()
        break


if __name__ == '__main__':
    main()
