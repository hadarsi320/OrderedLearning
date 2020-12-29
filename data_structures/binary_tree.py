import torch


class BinaryTree:
    """
    A full binary tree of binarized vectors
    For node in depth i, its left son contains a subset of its items that have 0 in the i+1-th cell
    For node in depth i, its right son contains a subset of its items that have 1 in the i+1-th cell
    """

    def __init__(self, data: torch.Tensor, tree_depth=None):
        assert all(item in [0, 1] for row in data for item in row)
        self._data = data
        if tree_depth is not None and tree_depth <= data.shape[1]:
            self._tree_depth = tree_depth
        else:
            self._tree_depth = data.shape[1]
        self._root = self.generate_tree()

    def generate_tree(self):
        initial_mask = torch.ones(self._data.shape[0], dtype=torch.bool)
        root = self.generate_node(initial_mask, -1, None)
        return root

    def generate_node(self, mask: torch.Tensor, depth, parent):
        node = BinaryTreeNode(mask, depth, parent)
        depth += 1
        if depth < self._tree_depth:
            left_son_mask = torch.logical_and(self._data[:, depth] == 0, mask)
            node.set_left_son(self.generate_node(left_son_mask, depth, node))
            right_son_mask = torch.logical_and(self._data[:, depth] == 1, mask)
            node.set_right_son(self.generate_node(right_son_mask, depth, node))
        return node

    def search_tree(self, query: torch.Tensor, result_size: int = None, depth: int = None):
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
    def __init__(self, data: torch.Tensor, depth, parent):
        self.index = depth
        self.data = data
        self.num_samples = self.data.shape[0]

        self.parent = parent
        self.left_son = None
        self.right_son = None

    def get_data(self):
        return self.data

    def get_parent(self):
        return self.parent

    def get_left_son(self):
        return self.left_son

    def get_right_son(self):
        return self.right_son

    def get_num_samples(self):
        return self.num_samples

    def is_leaf(self):
        return self.left_son is None and self.right_son is None

    def set_left_son(self, node):
        self.left_son = node

    def set_right_son(self, node):
        self.right_son = node
