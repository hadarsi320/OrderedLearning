import math

import torch


class BinaryTreeNode:
    def __init__(self, data: torch.Tensor, depth, parent=None, left_son=None, right_son=None):
        self.index = depth
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

    def __init__(self, data: torch.Tensor, data_repr: torch.Tensor, tree_depth=float('inf')):
        assert torch.all(torch.logical_or(torch.eq(data_repr, 0), torch.eq(data_repr, 1))) and \
               data_repr.shape[0] == data.shape[0]

        self.__data = data.cpu()
        self.__data_repr = data_repr.cpu()
        self._tree_depth = min(tree_depth, data_repr.shape[1])

        self._num_nodes = 0
        self._root = self._generate_tree()

    def _generate_tree(self):
        initial_mask = torch.ones(self.__data_repr.shape[0], dtype=torch.bool)
        root = self._generate_node(initial_mask, -1, None)
        return root

    def _generate_node(self, mask: torch.Tensor, depth, parent):
        self._num_nodes += 1
        if math.log2(self._num_nodes) == 0:
            print(self._num_nodes)
        node = BinaryTreeNode(mask, depth, parent)
        depth += 1
        if depth < self._tree_depth:
            left_son_mask = torch.logical_and(self.__data_repr[:, depth] == 0, mask)
            if torch.any(left_son_mask):
                node.left_son = self._generate_node(left_son_mask, depth, node)

            right_son_mask = torch.logical_and(self.__data_repr[:, depth] == 1, mask)
            if torch.any(right_son_mask):
                node.right_son = self._generate_node(right_son_mask, depth, node)
        return node

    def search_tree(self, item_repr: torch.Tensor, result_size: int = None, depth: int = None):
        node: BinaryTreeNode = self._root
        for i, unit in enumerate(item_repr.squeeze()):
            if torch.sum(node.data) <= 1 or \
                    (depth is not None and i == depth) or \
                    (result_size is not None and torch.sum(node.data) <= result_size):
                break

            if unit == 0:
                node = node.left_son
            else:
                node = node.right_son
        return self.__data[node.data]
