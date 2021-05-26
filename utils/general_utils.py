import time
from datetime import datetime

import torch
import matplotlib.pyplot as plt

from utils import to_rgb


def restore_image(image, mean, std, im_format='RGB') -> torch.tensor:
    std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1)
    mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1)
    restored_image = std * image + mean
    return to_rgb(restored_image, im_format).permute(1, 2, 0)


def plot_image(image):
    plt.imshow(image.permute(1, 2, 0))
    plt.show()


def binarize_data(data: torch.Tensor, bin_quantile=0.5):
    if not 0 < bin_quantile < 1:
        raise ValueError('The binarization quantile must be in range (0, 1)')

    binarized = torch.zeros_like(data)
    q_quantiles = torch.quantile(data, bin_quantile, dim=0)
    binarized[data >= q_quantiles] = 1
    return binarized


def get_current_time():
    return datetime.now().strftime('%y-%m-%d--%H-%M-%S')


def get_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() == 8:
            return torch.device('cuda:7')
        return torch.device('cuda')
    return torch.device('cpu')


def format_number(number, precision=3):
    if number == 0:
        return number

    if round(number, ndigits=precision) != 0:
        return round(number, ndigits=precision)

    return f'{number:.{precision}e}'


def format_time(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))


def plot_subfigures(images, cmap=None, title=None, figsize_scale=1):
    output_shape = images.shape[:2]
    fig, axes_mat = plt.subplots(*output_shape, figsize=(output_shape[1] * figsize_scale,
                                                         output_shape[0] * figsize_scale))
    for i, (image_row, axes) in enumerate(zip(images, axes_mat)):
        for j, (image, axis) in enumerate(zip(image_row, axes)):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.imshow(image, cmap=cmap)
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
