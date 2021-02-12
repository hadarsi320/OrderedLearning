import pickle
from time import time

import numpy as np
import torch

from data_structures.binary_tree import BinaryTree
from nueral_networks.autoencoders import Autoencoder
from utils_package import utils
from data import cifar10

import matplotlib.pyplot as plt


def linear_scan(item, data, repr, output_size=10) -> torch.Tensor:
    distance = {}
    for i, item_ in enumerate(repr):
        distance[i] = torch.sum(torch.abs(item - item_))
    ordered_neighbors = sorted(distance, key=lambda k: distance[k], reverse=True)
    return data[ordered_neighbors[:output_size]]


def eval_retrieval_method(data_repr: torch.Tensor, retrieval_method):
    times = []
    for i in range(data_repr.shape[1]):
        _times = []
        for sample in data_repr:
            start = time()
            retrieval_method(sample, i+1)
            _times.append(time() - start)
        times.append(np.average(_times))
    return times


def main():
    model_pickle = 'models/nestedDropoutAutoencoder_deep_ReLU_21-01-07__01-18-13.pkl'
    binary_tree_pickle = 'pickles/binary_tree_50.pkl'

    dataloader = cifar10.get_cifar10_dataloader(1000)
    device = utils.get_device()
    autoencoder: Autoencoder = torch.load(model_pickle, map_location=device)
    data = torch.cat([sample for sample, _ in dataloader])
    data_repr = utils.get_data_representation(autoencoder, dataloader, device).cpu()
    binarized_repr = utils.binarize_data(data_repr, bin_quantile=0.2)

    binary_tree: BinaryTree = pickle.load(open(binary_tree_pickle, 'rb'))

    def linear_scan_i(sample, i):
        return linear_scan(sample[:i], data, binarized_repr[:, i])
    linear_scan_times = eval_retrieval_method(binarized_repr, linear_scan_i)

    def tree_search_i(sample, i):
        return binary_tree.search_tree(sample, depth=i)
    tree_search_times = eval_retrieval_method(binarized_repr, tree_search_i)

    pickle.dump((linear_scan_times, tree_search_times), open(f'pickles/search_times_{utils.current_time()}.pkl', 'wb'))

    plt.plot(range(1, autoencoder.repr_dim + 1), linear_scan_times, label='Linear Scan')
    plt.plot(range(1, autoencoder.repr_dim + 1), tree_search_times, label='Tree Search')
    plt.xlabel('Code Length')
    plt.ylabel('Average retrieval time per query')
    plt.title('Retrieval time per code length')
    plt.legend()
    plt.savefig('plots/retrieval_time')


if __name__ == '__main__':
    main()
