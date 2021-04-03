import pickle
from datetime import timedelta
from time import time

import torch

from models.autoencoders import Autoencoder
import utils
from data import cifar10


class BinaryTreeNode:
    def __init__(self, data: torch.Tensor, depth, parent=None, left_son=None, right_son=None):
        self.depth = depth
        self.data = data

        self.parent = parent
        self.left_son = left_son
        self.right_son = right_son

    def is_leaf(self):
        return self.left_son is None and self.right_son is None

    def set_left_son(self, node):
        self.left_son = node

    def set_right_son(self, node):
        self.right_son = node


class BinaryTree:
    """
    A full binary tree of binarized vectors
    For node in depth i, its left son contains a subset of its items that have 0 in the i+1-th cell
    For node in depth i, its right son contains a subset of its items that have 1 in the i+1-th cell
    """

    def __init__(self, data: torch.Tensor, code: torch.Tensor, tree_depth=float('inf')):
        assert torch.all(torch.logical_or(torch.eq(code, 0), torch.eq(code, 1))) and \
               code.shape[0] == data.shape[0]

        self.__data = data.cpu()
        self.__code = code.cpu()
        self._tree_depth = min(tree_depth, code.shape[1])

        self._num_nodes = 0
        self._root = self._generate_tree()

    def _generate_tree(self):
        initial_mask = torch.ones(self.__code.shape[0], dtype=torch.bool)
        root = self._generate_node(initial_mask, -1, None)
        return root

    def _generate_node(self, mask, depth, parent):
        self._num_nodes += 1
        if utils.is_power_of_2(self._num_nodes):
            print(f'Creating node {self._num_nodes:,} out of {2 ** (self._tree_depth + 1) - 1:,}')

        node = BinaryTreeNode(mask, depth, parent)
        depth += 1
        if depth < self._tree_depth and torch.sum(mask) > 1:
            left_son_mask = torch.logical_and(self.__code[:, depth] == 0, mask)
            if torch.any(left_son_mask):
                node.left_son = self._generate_node(left_son_mask, depth, node)

            right_son_mask = torch.logical_and(self.__code[:, depth] == 1, mask)
            if torch.any(right_son_mask):
                node.right_son = self._generate_node(right_son_mask, depth, node)
        return node

    def search_tree(self, item: list, result_size: int = None, max_depth: int = None):
        if not isinstance(item, list):
            item = list(item)

        node: BinaryTreeNode = self._root
        for i, unit in enumerate(item):
            if i == self._tree_depth or len(node.data) == 1 or \
                    (max_depth is not None and i == max_depth) or \
                    (result_size is not None and len(node.data) <= result_size):
                break

            if unit == 0:
                if node.left_son is None:
                    break
                node = node.left_son

            else:
                if node.right_son is None:
                    break
                node = node.right_son

        return self.__data[node.data]

    def get_num_nodes(self):
        return self._num_nodes

    def get_depth(self):
        return self._tree_depth


def main():
    depth = 64
    bin_quantile = 0.2
    model_pickle = f'models/nestedDropoutAutoencoder_deep_ReLU_21-01-07__01-18-13.pkl'

    dataloader = cifar10.get_dataloader(download=True)
    device = utils.get_device()
    data = cifar10.load_data(dataloader)
    print('Data loaded')

    autoencoder: Autoencoder = torch.load(model_pickle, map_location=device)
    autoencoder.eval()
    print('Model loaded')

    representation = utils.get_data_representation(autoencoder, dataloader, device)
    del autoencoder
    data_repr = utils.binarize_data(representation, bin_quantile).cpu()
    print('Data representation created')

    binary_tree = BinaryTree(data, data_repr, tree_depth=depth)
    print(f'Binary tree created, with {binary_tree.get_num_nodes():,} nodes')
    pickle.dump({'binary tree': binary_tree, 'data_repr': data_repr},
                open(f'pickles/binary_tree_{depth}.pkl', 'wb'))
    print('The binary tree has been saved')


if __name__ == '__main__':
    start_time = time()
    main()
    print(f'Total run time: {timedelta(seconds=time() - start_time)}')
