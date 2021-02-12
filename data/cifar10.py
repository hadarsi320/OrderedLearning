import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

from .constants import CIFAR10_MEAN, CIFAR10_STD


def get_cifar10_dataloader(batch_size=1):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                                    transforms.Lambda(lambda t: t.view(-1))])
    dataset = torchvision.datasets.CIFAR10(root='data/', download=True,
                                           transform=transform, train=True)
    if batch_size == -1:
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def load_cifar10(dataloader=None):
    if dataloader is None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                                        transforms.Lambda(lambda t: t.view(-1))])
        dataset = torchvision.datasets.CIFAR10(root='data/', download=True,
                                               transform=transform, train=True)
        return torch.stack([sample for sample, _ in dataset])

    else:
        return torch.cat([sample for sample, _ in dataloader])
