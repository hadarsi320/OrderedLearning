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
    mode The mode of product to compute, either 'all pairs' or 'serial'

    Returns
    -------
    The product of the filters
    """
    if weights.shape[1] != 1:
        raise NotImplementedError('Filter products only implemented for convolutions with input dim 1')

    if mode == 'all pairs':
        product = weights.matmul(weights.squeeze().transpose(1, 2))

    elif mode == 'serial':
        product = weights[:-1].squeeze() @ weights[1:].squeeze().transpose(1, 2)

    else:
        raise ValueError('Mode must be either "all pairs" or "serial"')

    return product.pow(2).sum().sqrt()
