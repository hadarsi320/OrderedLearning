import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

CIFAR10_MEAN = [0.49139968, 0.48215841, 0.44653091]
CIFAR10_STD = [0.24703223, 0.24348513, 0.26158784]


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
    return dataset, dataloader


def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                                    transforms.Lambda(lambda t: t.view(-1))])
    dataset = torchvision.datasets.CIFAR10(root='data/', download=True,
                                           transform=transform, train=True)
    return torch.stack([sample for sample, _ in dataset])
