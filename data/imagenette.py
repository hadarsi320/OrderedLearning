import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils
from data.constants import CIFAR10_MEAN, CIFAR10_STD


def get_dataloader(batch_size=1, normalize=True, image_mode='RGB'):
    transform_list = []
    if image_mode == 'YCbCr':
        transform_list = [transforms.Lambda(lambda image: image.convert('YCbCr'))]
    transform_list.append(transforms.ToTensor())
    if normalize:
        if image_mode == 'RGB':
            pass
        elif image_mode == 'YCbCr':
            pass
        else:
            NotImplementedError(f'Image mode {image_mode} not implemented')

    transform = transforms.Compose(transform_list)
    dataset = torchvision.datasets.ImageFolder(utils.imagenette_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def load_data(dataloader=None, normalize=True):
    if dataloader is None:
        dataloader = get_dataloader(normalize=normalize)
    return torch.cat([sample for sample, _ in dataloader])


def unnormalize(im: torch.Tensor):
    return utils.restore_image(im.cpu().view(3, 32, 32), CIFAR10_MEAN, CIFAR10_STD)
