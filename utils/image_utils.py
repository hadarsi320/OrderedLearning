import PIL.Image
import torch
from torchvision.transforms import functional as F

import utils


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    rgb_image = F.to_pil_image(image, mode='RGB')
    ycbcr_image = rgb_image.convert('YCbCr')
    ycbcr_tensor = F.to_tensor(ycbcr_image)
    return ycbcr_tensor

    # if not isinstance(image, torch.Tensor):
    #     raise TypeError("Input type is not a torch.Tensor. Got {}".format(
    #         type(image)))
    #
    # if len(image.shape) < 3 or image.shape[-3] != 3:
    #     raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
    #                      .format(image.shape))
    #
    # r: torch.Tensor = image[..., 0, :, :]
    # g: torch.Tensor = image[..., 1, :, :]
    # b: torch.Tensor = image[..., 2, :, :]
    #
    # delta: float = 0.5
    # y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    # cb: torch.Tensor = (b - y) * 0.564 + delta
    # cr: torch.Tensor = (r - y) * 0.713 + delta
    # return torch.stack([y, cb, cr], -3)


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    ycbcr_image = F.to_pil_image(image, mode='YCbCr')
    rgb_image = ycbcr_image.convert('RGB')
    rgb_tensor = F.to_tensor(rgb_image)
    return rgb_tensor

    # if not isinstance(image, torch.Tensor):
    #     raise TypeError("Input type is not a torch.Tensor. Got {}".format(
    #         type(image)))
    #
    # if len(image.shape) < 3 or image.shape[-3] != 3:
    #     raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
    #                      .format(image.shape))
    #
    # y: torch.Tensor = image[..., 0, :, :]
    # cb: torch.Tensor = image[..., 1, :, :]
    # cr: torch.Tensor = image[..., 2, :, :]
    #
    # delta: float = 0.5
    # cb_shifted: torch.Tensor = cb - delta
    # cr_shifted: torch.Tensor = cr - delta
    #
    # r: torch.Tensor = y + 1.403 * cr_shifted
    # g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    # b: torch.Tensor = y + 1.773 * cb_shifted
    # return torch.stack([r, g, b], -3)


def y_to_rgb(image: torch.Tensor):
    assert image.shape[-3] == 1

    shape = list(image.shape)
    shape[-3] = 3
    new_image = torch.zeros(shape)

    new_image[..., 0, :, :] = image
    new_image[..., 1, :, :] = 0.5
    new_image[..., 2, :, :] = 0.5
    return ycbcr_to_rgb(new_image)


def to_rgb(image, im_format):
    if im_format == 'RGB':
        return image
    elif im_format == 'Y':
        return utils.y_to_rgb(image)
    elif im_format == 'YCbCr':
        return utils.ycbcr_to_rgb(image)
    else:
        raise NotImplementedError()
