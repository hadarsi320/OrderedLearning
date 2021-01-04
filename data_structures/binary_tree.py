import torch


class BinaryTreeNode:
    def __init__(self, data: torch.Tensor, depth, parent=None, left_son=None, right_son=None):
        self.index = depth
        self.data = data
        self.num_samples = self.data.shape[0]

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

    def __init__(self, data: torch.Tensor, tree_depth=None):
        # TODO implement the addition of both data and its binary encoding, such that the tree will return the actual
        #  data and not just a mask over it
        assert all(item in [0, 1] for row in data for item in row)
        self._data = data
        if tree_depth is not None and tree_depth <= data.shape[1]:
            self._tree_depth = tree_depth
        else:
            self._tree_depth = data.shape[1]
        self._root = self._generate_tree()

    def _generate_tree(self):
        initial_mask = torch.ones(self._data.shape[0], dtype=torch.bool)
        root = self._generate_node(initial_mask, -1, None)
        return root

    def _generate_node(self, mask: torch.Tensor, depth, parent):
        node = BinaryTreeNode(mask, depth, parent)
        depth += 1
        if depth < self._tree_depth:
            left_son_mask = torch.logical_and(self._data[:, depth] == 0, mask)
            if torch.any(left_son_mask):
                node.left_son = self._generate_node(left_son_mask, depth, node)

            right_son_mask = torch.logical_and(self._data[:, depth] == 1, mask)
            if torch.any(right_son_mask):
                node.right_son = self._generate_node(right_son_mask, depth, node)
        return node

    def search_tree(self, query: torch.Tensor, result_size: int = None, depth: int = None):
        node: BinaryTreeNode = self._root
        for i, item in enumerate(query.squeeze()):
            if depth is not None and i == depth:
                break
            if result_size is not None and node.num_samples <= result_size:
                break

            if item == 0:
                node = node.left_son
            else:
                node = node.right_son
        return node.data
