import pickle

import torch
from torchvision.transforms import ToPILImage

from data_structures.binary_tree import BinaryTree
from nueral_networks.autoencoders import Autoencoder
from utils_package import data_utils, utils

import matplotlib.pyplot as plt

import random


def main():
    depth = 15
    bin_quantile = 0.2
    model_pickle = f'models/nestedDropoutAutoencoder_deep_deep_ReLU_21-01-07__01-18-13.pkl'

    dataset, dataloader = data_utils.load_cifar10(-1)
    autoencoder: Autoencoder = torch.load(open(model_pickle, 'rb'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    autoencoder.to(device)
    autoencoder.eval()

    data, _ = next(iter(dataloader))
    print('data loaded')

    representation = utils.get_data_representation(autoencoder, dataloader, device)
    data_repr = utils.binarize_data(representation, bin_quantile)
    print('data representation created')

    binary_tree = BinaryTree(data, data_repr, tree_depth=depth)
    pickle.dump(binary_tree, open(f'binary_tree_{depth}', 'wb'))
    print(f'Binary tree created, with {binary_tree.get_num_nodes()} nodes')

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
