import torch


class BinaryTree:
    """
    A full binary tree of binarized vectors
    For node in depth i, its left son contains a subset of its items that have 0 in the i+1-th cell
    For node in depth i, its right son contains a subset of its items that have 1 in the i+1-th cell
    """

    def __init__(self, data: torch.tensor):
        assert all(item in [0, 1] for row in data for item in row)
        self._data = data
        self._tree_depth = data.shape[1]
        self._root = self.generate_node(self._data, depth=-1)

    def generate_node(self, data, depth, parent=None):
        node = BinaryTreeNode(data, depth, parent)
        depth += 1
        if depth < self._tree_depth:
            left_son_data = data[data[:, depth] == 0]
            node.set_left_son(self.generate_node(left_son_data, depth, node))
            right_son_data = data[data[:, depth] == 1]
            node.set_right_son(self.generate_node(right_son_data, depth, node))
        return node

    def search_tree(self, query: torch.tensor, result_size: int = None, depth: int = None):
        node = self._root
        for i, item in enumerate(query.squeeze()):
            if depth is not None and i == depth:
                break
            if result_size is not None and node.get_num_samples() <= result_size:
                break

            if item == 0:
                node = node.get_left_son()
            else:
                node = node.get_right_son()

        return node.get_data()


class BinaryTreeNode:
    def __init__(self, data: torch.tensor, index, parent):
        self._index = index
        self._data = data
        self._num_samples = self._data.shape[0]

        self._parent = parent
        self._left_son = None
        self._right_son = None

    def get_data(self):
        return self._data

    def get_parent(self):
        return self._parent

    def get_left_son(self):
        return self._left_son

    def get_right_son(self):
        return self._right_son

    def get_num_samples(self):
        return self._num_samples

    def is_leaf(self):
        return self._left_son is None and self._right_son is None

    def set_left_son(self, node):
        self._left_son = node

    def set_right_son(self, node):
        self._right_son = node
