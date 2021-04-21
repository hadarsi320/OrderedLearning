import matplotlib.pyplot as plt
import torch

import utils
from data import cifar10
from models.autoencoders import *


@torch.no_grad()
def compare_code_lengths():
    model_pickle = 'models/nestedDropoutAutoencoder_shallow_ReLU_21-01-13__02-50-53_dict.pt'
    torch.random.manual_seed(42)
    num_images = 6
    code_lengths = [16, 64, 128, 256, 1024, 'Original']

    device = utils.get_device()
    model: Autoencoder = torch.load(model_pickle, map_location=device)['autoencoder']
    model.eval()

    dataset = cifar10.get_dataloader().dataset

    plt.tight_layout()
    fig, axes = plt.subplots(ncols=num_images, nrows=len(code_lengths), squeeze=False, figsize=(12, 12))
    indices = torch.randint(len(dataset), (num_images,))
    for i, index in enumerate(indices):
        original_image, _ = dataset[index]
        encoding = model.encode(original_image.to(device))
        for code_length, axis in zip(code_lengths, axes):
            if code_length == 'Original':
                image = original_image
            else:
                if code_length == 'Full':
                    encoding_ = encoding
                else:
                    encoding_ = torch.zeros_like(encoding)
                    encoding_[:code_length] = encoding[:code_length]
                image = model.decode(encoding_)
            axis[i].imshow(cifar10.unnormalize(image))
            axis[i].set_xticks([])
            axis[i].set_yticks([])
            if i == 0:
                axis[i].set_ylabel(code_length, fontsize=16)

    plt.show()


@torch.no_grad()
def compare_num_channels():
    model_dir = '/mnt/ml-srv1/home/hadarsi/ordered_learning/saves/cae-C-NestedDropoutAutoencoder_21-04-20--16-55-48/'
    torch.random.manual_seed(42)
    num_images = 6
    num_channels = [1, 2, 4, 8, 32, 'Original']

    save_dict = torch.load(f'{model_dir}/model.pt')
    normalized = save_dict['normalize_data']

    model = NestedDropoutAutoencoder(ConvAutoencoder(**save_dict), **save_dict)
    model.load_state_dict(save_dict['model'])
    model.eval()
    images, _ = next(iter(cifar10.get_dataloader(num_images, normalize=normalized)))

    plt.tight_layout()
    fig, axes_mat = plt.subplots(ncols=num_images, nrows=len(num_channels), squeeze=False, figsize=(12, 12))
    for i, (original_image, axes) in enumerate(zip(images, axes_mat.transpose())):
        encoding = model.encode(original_image).squeeze()
        for n, axis in zip(num_channels, axes):
            if n == 'Original':
                image = original_image
            else:
                encoding_ = torch.zeros_like(encoding)
                encoding_[:n] = encoding[:n]
                image = model.decode(encoding_).squeeze()
            axis.imshow(cifar10.unnormalize(image))
            axis.set_xticks([])
            axis.set_yticks([])
            if i == 0:
                axis.set_ylabel(n, fontsize=16)
    plt.show()


@torch.no_grad()
def reconstruct_images():
    torch.random.manual_seed(52)
    num_images = 8
    model_dir = '/mnt/ml-srv1/home/hadarsi/ordered_learning/saves/cae-C-ConvAutoencoder_21-04-20--16-43-34/'

    save_dict = torch.load(f'{model_dir}/model.pt')
    normalized = save_dict['normalize_data']

    cae = ConvAutoencoder(save_dict['mode'], dim=32, normalize_data=normalized)
    cae.load_state_dict(save_dict['model'])
    cae.eval()

    images, _ = next(iter(cifar10.get_dataloader(num_images, normalize=normalized)))
    fig, axes_mat = plt.subplots(ncols=num_images, nrows=2,
                                 gridspec_kw={'wspace': 0, 'hspace': 0})

    for i, axes in enumerate(axes_mat):
        for image, axis in zip(images, axes):
            # axis.axis('off')
            axis.set_xticks([])
            axis.set_yticks([])
            if i == 1:
                image = cae(image.unsqueeze(0)).squeeze()
            if normalized:
                image = cifar10.unnormalize(image)
            else:
                image = image.permute(1, 2, 0)
            axis.imshow(image)
    plt.show()


if __name__ == '__main__':
    compare_num_channels()
