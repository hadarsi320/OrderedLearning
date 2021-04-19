import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .constants import CIFAR10_MEAN, CIFAR10_STD
import utils


def get_dataloader(batch_size=1, download=False, normalize=True):
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))

    transform = transforms.Compose(transform_list)
    dataset = torchvision.datasets.CIFAR10(root=utils.CIFAR10_file, download=download, transform=transform, train=True)
    if batch_size == -1:
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def load_data(dataloader=None, normalize=True):
    if dataloader is None:
        dataloader = get_dataloader(normalize=normalize)
    return torch.cat([sample for sample, _ in dataloader])


def unnormalize(im: torch.Tensor):
    return utils.restore_image(im.cpu().view(3, 32, 32), CIFAR10_MEAN, CIFAR10_STD)
