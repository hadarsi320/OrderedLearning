import torch
from matplotlib import pyplot as plt

from .image_utils import to_rgb


def restore_image(image, mean, std, im_format='RGB') -> torch.tensor:
    std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1)
    mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1)
    restored_image = std * image + mean
    return to_rgb(restored_image, im_format).permute(1, 2, 0)


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
