import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils
from data.constants import IMAGENETTE_Y_MEAN, IMAGENETTE_Y_STD


def get_dataloader(train=True, batch_size=1, normalize=True, image_mode='Y'):
    data_dir = utils.imagenette_train_dir if train else utils.imagenette_eval_dir
    transform_list = []
    if image_mode in ['Y', 'YCbCr']:
        transform_list.append(transforms.Lambda(lambda image: image.convert('YCbCr')))
    transform_list.extend([transforms.Resize((224, 224)),
                           transforms.ToTensor()])
    if image_mode == 'Y':
        transform_list.append(transforms.Lambda(lambda image: image[[0]]))

    if normalize:
        if image_mode == 'Y':
            transform_list.append(transforms.Normalize(IMAGENETTE_Y_MEAN, IMAGENETTE_Y_STD))
        else:
            NotImplementedError(f'Normalization for image mode {image_mode} not implemented')

    transform = transforms.Compose(transform_list)
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def load_data(dataloader=None, normalize=True):
    if dataloader is None:
        dataloader = get_dataloader(normalize=normalize)
    return torch.cat([sample for sample, _ in dataloader])


def unnormalize(im: torch.Tensor, im_format):
    if im.dim() == 2:
        im = im.unsqueeze(0)
    if im.shape == (1, 224, 224):
        return utils.restore_image(im, IMAGENETTE_Y_MEAN, IMAGENETTE_Y_STD, im_format)
    else:
        raise NotImplementedError('Unnormalization not implemented for said shape')
