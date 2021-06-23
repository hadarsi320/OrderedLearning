import math

import torch


def is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


def get_power_successor(n, base=2):
    """
    Finds the smallest power of 2 which is larger than n
    """
    if math.log(n, base).is_integer():
        return n * base
    return pow(base, math.ceil(math.log(n, base)))


def square_shape(x, y):
    """Approximated square proportions"""
    total = x * y
    div = math.ceil(total ** 0.5)
    while total % div != 0:
        div += 1
    return div, total // div


def filters_product(weights: torch.Tensor, mode: str = 'all pairs') -> torch.Tensor:
    """
    Parameters
    ----------
    weights The weights of a convolutional layer with input dim of 1
    mode The mode of product to compute, either 'all pairs' or 'serial' or 'row col'

    Returns
    -------
    The product of the filters
    """
    if weights.shape[1] != 1:
        raise NotImplementedError('Filter products only implemented for convolutions with input dim 1')

    if mode == 'all pairs':
        product = weights.matmul(weights.squeeze().transpose(1, 2))
        product_sum = product.pow(2).sum().sqrt()

    elif mode == 'serial':
        product = weights[:-1].squeeze() @ weights[1:].squeeze().transpose(1, 2)
        product_sum = product.pow(2).sum().sqrt()

    elif mode == 'row col':
        if weights.shape != (64, 1, 8, 8):
            raise NotImplementedError('row col mode only implemented for dct type convolutions')
        weights = weights.reshape(8, 8, 8, 8)
        product_sum = 0
        for i in range(8):
            row_prod = weights[i].unsqueeze(1) @ weights[i].transpose(-2, -1)
            col_prod = weights[:, i].unsqueeze(1).transpose(-2, -1) @ weights[:, i]

            product_sum += row_prod.pow(2).sum()
            product_sum += col_prod.pow(2).sum()
        product_sum = product_sum.sqrt()

    else:
        raise ValueError('Mode must be either "all pairs", "serial" or "row col"')

    return product_sum
