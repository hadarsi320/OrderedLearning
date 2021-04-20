import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils


def get_dataloader(batch_size=1, download=False, normalize=False):
    transform_list = [transforms.ToTensor(), transforms.Resize((64, 64))]
    if normalize:
        raise NotImplementedError('Normalization is yet to be implemented for celeba')

    transform = transforms.Compose(transform_list)
    dataset = torchvision.datasets.CelebA(root=utils.datasets_dir, download=download, transform=transform, split='all')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def load_data(dataloader=None, normalize=True):
    if dataloader is None:
        dataloader = get_dataloader(normalize=normalize)
    return torch.cat([sample for sample, _ in dataloader])


def unnormalize(im: torch.Tensor):
    raise NotImplementedError('Normalization is yet to be implemented for celeba')
    # return utils.restore_image(im.cpu().view(3, 32, 32), CIFAR10_MEAN, CIFAR10_STD)
