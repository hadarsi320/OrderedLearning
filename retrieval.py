import math
import pickle
from collections import defaultdict
from time import time

import numpy as np
import torch
from tqdm import tqdm

from binary_tree import BinaryTree, BinaryTreeNode
from utils_package import data_utils, utils

import matplotlib.pyplot as plt


def linear_scan(item, data, coded_data, output_size=10) -> torch.Tensor:
    distance = defaultdict(int)
    for i, item_ in enumerate(coded_data):
        for j in range(len(item)):
            if item_[j] != item[j]:
                distance[i] += 1
    ordered_neighbors = sorted(distance, key=lambda k: distance[k], reverse=True)
    return data[ordered_neighbors[:output_size]]


def evaluate_retrieval_method(data_repr: torch.Tensor, retrieval_method, code_length, num_samples=1000) -> dict:
    times = {}
    for i in tqdm(range(math.ceil(math.log2(code_length) + 1))):
        i = 2 ** i if 2 ** i <= code_length else code_length

        _times = []
        if num_samples is None:
            samples = data_repr
        else:
            samples = data_repr[torch.randint(len(data_repr), (num_samples,))]

        # for sample in tqdm(samples, desc=f'Code length {i}'):
        for sample in samples:
            start = time()
            retrieval_method(sample, i)
            _times.append(time() - start)
        times[i] = np.average(_times)
    return times


def main():
    binary_tree_pickle = 'pickles/binary_tree_32.pkl'
    current_time = utils.current_time()

    data = data_utils.load_cifar10()
    print('Data loaded')

    pickle_dict = pickle.load(open(binary_tree_pickle, 'rb'))
    binary_tree = pickle_dict['binary tree']
    binarized_repr = pickle_dict['data_repr']
    print('Binary tree loaded')

    repr_dim = min(binarized_repr.shape[1], binary_tree.get_depth())

    #       binary tree retrieval
    def tree_search_i(sample, i):
        return binary_tree.search_tree(list(sample)[:i], max_depth=i)

    tree_search_times = evaluate_retrieval_method(binarized_repr, tree_search_i, repr_dim)
    pickle.dump(tree_search_times, open(f'pickles/or_retrieval_times_{current_time}.pkl', 'wb'))

    #       linear scan
    def linear_scan_i(sample, i):
        binarized_repr_i = binarized_repr[:, i].view(len(binarized_repr), -1)
        return linear_scan(sample[:i], data, binarized_repr_i)

    linear_scan_times = evaluate_retrieval_method(binarized_repr, linear_scan_i, repr_dim)
    pickle.dump(linear_scan_times, open(f'pickles/ls_retrieval_times_{current_time}.pkl', 'wb'))

    # plotting
    plt.plot(tree_search_times.keys(), tree_search_times.values(), label='Tree Search')
    plt.plot(linear_scan_times.keys(), linear_scan_times.values(), label='Linear Scan')
    plt.xlabel('Code Length')
    plt.ylabel('Average retrieval time per query')
    plt.title('Retrieval time per code length')
    plt.legend()
    plt.savefig('plots/retrieval_times')


if __name__ == '__main__':
    main()
