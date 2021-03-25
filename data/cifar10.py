import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .constants import CIFAR10_MEAN, CIFAR10_STD
import utils


def get_dataloader(batch_size=1, download=False):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                                    transforms.Lambda(lambda t: t.view(-1))])
    dataset = torchvision.datasets.CIFAR10(root='./data/', download=download, transform=transform, train=True)
    # dataset = torchvision.datasets.CIFAR10(root='/mnt/ml-srv1/home/hadarsi/ordered_learning/data/',
    #                                        download=False, transform=transform, train=True)
    if batch_size == -1:
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def load_data(dataloader=None):
    if dataloader is None:
        dataloader = get_dataloader()
    return torch.cat([sample for sample, _ in dataloader])


def restore(im: torch.Tensor):
    return utils.restore_image(im.cpu().view(3, 32, 32), CIFAR10_MEAN, CIFAR10_STD)