import math
from math import cos, pi

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


def filter_correlation(weights: torch.Tensor, mode: str = 'all pairs') -> torch.Tensor:
    """
    Parameters
    ----------
    weights The weights of a convolutional layer with input dim of 1
    mode The mode of product to compute

    Returns
    -------
    The correlation between the filters (products)
    """
    if weights.shape[1] != 1:
        raise NotImplementedError('Filter products only implemented for convolutions with input dim 1')

    # if mode == 'all pairs':
    #     product = weights @ weights.squeeze().transpose(1, 2)
    #     product_sum = product.pow(2).sum().sqrt()
    #
    # elif mode == 'serial':
    #     product = weights[:-1].squeeze() @ weights[1:].squeeze().transpose(1, 2)
    #     product_sum = product.pow(2).sum().sqrt()
    #
    # elif mode == 'row col':
    #     if weights.shape != (64, 1, 8, 8):
    #         raise NotImplementedError('row col mode only implemented for dct type convolutions')
    #     weights = weights.reshape(8, 8, 8, 8)
    #     product_sum = 0
    #     for i in range(8):
    #         row_prod = weights[i].unsqueeze(1) @ weights[i].transpose(-2, -1)
    #         col_prod = weights[:, i].unsqueeze(1).transpose(-2, -1) @ weights[:, i]
    #
    #         product_sum += row_prod.pow(2).sum()
    #         product_sum += col_prod.pow(2).sum()
    #     product_sum = product_sum.sqrt()

    elif mode == 'hadamund':
        product = weights * weights.squeeze()
        for i in range(product.shape[0]):
            product[i, i] = 0
        product_sum = product.sum().pow(2)

    elif mode == 'frobenius':
        product = weights @ weights.squeeze().transpose(1, 2)
        for i in range(product.shape[0]):
            product[i, i] = 0
        product_sum = product.sum((-2, -1)).pow(2).sum()

    else:
        raise ValueError('Illegal mode given, must be "frobenius" or "hadamund"')

    return product_sum


def generate_dct_filters() -> torch.Tensor:
    tensor = torch.empty(8, 8, 8, 8)
    for k1 in range(8):
        for k2 in range(8):
            for n1 in range(8):
                for n2 in range(8):
                    tensor[k1, k2, n1, n2] = cos(pi / 8 * (n2 + 1 / 2) * k2) * \
                                             cos(pi / 8 * (n1 + 1 / 2) * k1)
    return tensor
