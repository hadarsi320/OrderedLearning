import math
import pickle
from collections import defaultdict
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import utils
from data_structures.binary_tree import BinaryTree
from data import cifar10


def linear_scan(item, data, coded_data, output_size=10) -> torch.Tensor:
    distance = defaultdict(int)
    for i, item_ in enumerate(coded_data):
        for j in range(len(item)):
            if item_[j] != item[j]:
                distance[i] += 1
    ordered_neighbors = sorted(distance, key=lambda k: distance[k], reverse=True)
    return data[ordered_neighbors[:output_size]]


def evaluate_retrieval_method(data_repr: torch.Tensor, retrieval_method, code_length, num_samples=100) -> dict:
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


def test_retrieval_times():
    binary_tree_pickle = 'pickles/binary_tree_64.pkl'
    current_time = utils.get_current_time()

    data = cifar10.load_data()
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
        binarized_repr_i = binarized_repr[:, :i].view(len(binarized_repr), -1)
        return linear_scan(sample[:i], data, binarized_repr_i)

    linear_scan_times = evaluate_retrieval_method(binarized_repr, linear_scan_i, repr_dim)
    pickle.dump(linear_scan_times, open(f'pickles/ls_retrieval_times_{current_time}.pkl', 'wb'))

    # tree_search_times = pickle.load(open('pickles/or_retrieval_times_21-01-14__16-18-02.pkl', 'rb'))
    # linear_scan_times = pickle.load(open('pickles/ls_retrieval_times_21-01-14__13-03-04.pkl', 'rb'))

    # plotting
    plt.plot(tree_search_times.keys(), tree_search_times.values(), label='Tree Search')
    plt.plot(linear_scan_times.keys(), linear_scan_times.values(), label='Linear Scan')
    plt.xlabel('Code Length')
    plt.ylabel('Average retrieval time per query')
    plt.title('Retrieval time per code length')
    plt.yticks(list(tree_search_times.keys()))
    plt.yscale('log')
    plt.legend()
    plt.savefig('plots/retrieval_times')
    plt.show()


def explore_query(binary_tree, query):
    for i in range(1, binary_tree.get_depth()):
        results = binary_tree.search_tree(list(query), max_depth=i)
        print(f'{i}: {len(results)}')
        if len(results) == 1:
            break


def retrieve_images():
    binary_tree_pickle = 'pickles/binary_tree_32.pkl'
    torch.random.manual_seed(1)

    pickle_dict = pickle.load(open(binary_tree_pickle, 'rb'))
    binary_tree: BinaryTree = pickle_dict['binary tree']
    binarized_repr = pickle_dict['data_repr']

    sample = binarized_repr[torch.randint(len(binarized_repr), (1,))].squeeze()
    explore_query(binary_tree, sample)

    depths = {10024: [13, 14, 19], 15243: [18, 19, 20]}
    for depth in depths[seed]:
        results = binary_tree.search_tree(list(sample), max_depth=depth)
        fig, axes = plt.subplots(ncols=len(results))
        fig.suptitle(f'Retrieval results at depth {depth}')
        for image, axis in zip(results, axes):
            axis.imshow(cifar10.unnormalize(image))
            axis.set_xticks([])
            axis.set_yticks([])
        plt.savefig(f'plots/retrieval_{seed}_{depth}')
        plt.show()

    image = binary_tree.search_tree(list(sample))[0]
    plt.imshow(cifar10.unnormalize(image))
    plt.title('The query image')
    plt.yticks([])
    plt.xticks([])
    plt.show()


if __name__ == '__main__':
    # retrieve_images()
    test_retrieval_times()
