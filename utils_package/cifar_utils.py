import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils_package import utils

CIFAR10_MEAN = [0.49139968, 0.48215841, 0.44653091]
CIFAR10_STD = [0.24703223, 0.24348513, 0.26158784]


def get_cifar10_dataloader(batch_size=1, download=False):
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


def load_cifar10(dataloader=None):
    if dataloader is None:
        dataloader = get_cifar10_dataloader()
    return torch.cat([sample for sample, _ in dataloader])


def restore(im: torch.Tensor):
    return utils.restore_image(im.cpu().view(3, 32, 32), CIFAR10_MEAN, CIFAR10_STD)
