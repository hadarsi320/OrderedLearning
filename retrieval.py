import pickle
from time import time

import numpy as np
import torch
from tqdm import tqdm

from data_structures.binary_tree import BinaryTree
from nueral_networks.autoencoders import Autoencoder
from utils_package import data_utils, utils

import matplotlib.pyplot as plt


def linear_scan(item, data, coded_data, output_size=10) -> torch.Tensor:
    distance = {}
    for i, item_ in enumerate(coded_data):
        distance[i] = torch.sum(torch.abs(item - item_))
    ordered_neighbors = sorted(distance, key=lambda k: distance[k], reverse=True)
    return data[ordered_neighbors[:output_size]]


def evaluate_retrieval_method(data_repr: torch.Tensor, retrieval_method, code_length, num_samples=100):
    times = []
    for i in tqdm(range(code_length)):
        _times = []
        samples = data_repr[torch.randint(len(data_repr), (num_samples,))]
        for sample in samples:
            start = time()
            retrieval_method(sample, i + 1)
            _times.append(time() - start)
        times.append(np.average(_times))
    return times


def main():
    model_pickle = 'models/nestedDropoutAutoencoder_deep_ReLU_21-01-07__01-18-13.pkl'
    binary_tree_pickle = 'pickles/binary_tree_50.pkl'
    current_time = utils.current_time()

    dataloader = data_utils.get_cifar10_dataloader(1000)
    device = utils.get_device()
    data = torch.cat([sample for sample, _ in dataloader])
    print('Data loaded')

    autoencoder: Autoencoder = torch.load(model_pickle, map_location=device)
    print('Autoencoder loaded')

    data_repr = utils.get_data_representation(autoencoder, dataloader, device).cpu()
    binarized_repr = utils.binarize_data(data_repr, bin_quantile=0.2)
    repr_dim = autoencoder.repr_dim
    del autoencoder
    print('Code created')

    # binary tree retrieval
    binary_tree: BinaryTree = pickle.load(open(binary_tree_pickle, 'rb'))
    repr_dim = min(repr_dim, binary_tree.get_depth())
    print('Binary tree loaded')

    def tree_search_i(sample, i):
        return binary_tree.search_tree(sample, depth=i)

    tree_search_times = evaluate_retrieval_method(binarized_repr[:100], tree_search_i, repr_dim)
    pickle.dump(tree_search_times, open(f'pickles/or_retrieval_times_{current_time}.pkl', 'wb'))

    # linear retrieval
    def linear_scan_i(sample, i):
        return linear_scan(sample[:i], data, binarized_repr[:, i])

    linear_scan_times = evaluate_retrieval_method(binarized_repr, linear_scan_i, repr_dim)
    pickle.dump(linear_scan_times, open(f'pickles/ls_retrieval_times_{current_time}.pkl', 'wb'))

    # plotting
    plt.plot(range(1, repr_dim + 1), linear_scan_times, label='Linear Scan')
    plt.plot(range(1, repr_dim + 1), tree_search_times, label='Tree Search')
    plt.xlabel('Code Length')
    plt.ylabel('Average retrieval time per query')
    plt.title('Retrieval time per code length')
    plt.legend()
    plt.savefig('plots/retrieval_times')


if __name__ == '__main__':
    main()
