import math


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
