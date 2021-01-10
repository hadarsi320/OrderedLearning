import math

import torch

from utils_package import math_utils


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
        # initial_mask = torch.ones(self.__data_repr.shape[0], dtype=torch.bool)
        root = self._generate_node(self.__data, self.__code, -1, None)
        return root

    def _generate_node(self, data: torch.Tensor, code: torch.Tensor, depth, parent):
        self._num_nodes += 1
        if math_utils.is_power_of_2(self._num_nodes):
            print(f'Creating node {self._num_nodes:,} out of {2**self._tree_depth:,}')

        node = BinaryTreeNode(data, depth, parent)
        depth += 1
        if depth < self._tree_depth:
            left_son_mask = code[:, depth] == 0
            if torch.any(left_son_mask):
                node.left_son = self._generate_node(data[left_son_mask], code[left_son_mask], depth, node)

            right_son_mask = code[:, depth] == 1
            if torch.any(right_son_mask):
                node.right_son = self._generate_node(data[right_son_mask], code[right_son_mask], depth, node)
        return node

    def search_tree(self, item_repr: torch.Tensor, result_size: int = None, depth: int = None):
        node: BinaryTreeNode = self._root
        for i, unit in enumerate(item_repr.squeeze()):
            if len(node.data) <= 1 or \
                    (depth is not None and i == depth) or \
                    (result_size is not None and len(node.data) <= result_size):
                break

            if unit == 0:
                node = node.left_son
            else:
                node = node.right_son
        return node.data

    def get_num_nodes(self):
        return self._num_nodes