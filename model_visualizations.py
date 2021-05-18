import math
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

import utils
from data import imagenette
from models import *
from utils import get_device, get_data_representation


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
def plot_repr_var(autoencoder, dataloader, scale='log', show=False, **kwargs):
    device = kwargs.get('device', get_device())
    autoencoder = autoencoder.to(device)

    plt.clf()
    reprs = get_data_representation(autoencoder, dataloader, device)
    if reprs.dim() == 4:
        plt.plot(torch.mean(torch.var(reprs, dim=[0]), dim=[1, 2]).cpu())
    else:
        plt.plot(torch.var(reprs, dim=0).cpu())
    plt.yscale(scale)
    plt.xlabel(kwargs.get('xlabel', 'Units'))
    plt.ylabel(kwargs.get('ylabel', 'Variance'))

    if 'title' in kwargs:
        plt.title(kwargs.pop('title'))

    if 'savefig' in kwargs:
        plt.savefig(kwargs.pop('savefig'))

    if show:
        plt.show()


@torch.no_grad()
def plot_images_by_channels(model, data_module, normalized, im_format='RGB'):
    torch.random.manual_seed(42)
    num_channels = [1, 2, 4, 8, 16, 32, 64, 'Original']
    num_images = len(num_channels)

    images, _ = next(iter(data_module.get_dataloader(num_images, normalize=normalized)))

    plt.tight_layout()
    fig, axes_mat = plt.subplots(ncols=num_images, nrows=len(num_channels), squeeze=False,
                                 figsize=(num_images * 2, num_images * 2))
    for i, (original_image, axes) in enumerate(zip(images, axes_mat.transpose())):
        encoding = model.encode(original_image).squeeze()
        for n, axis in zip(num_channels, axes):
            if n == 'Original':
                image = original_image
            else:
                encoding_ = torch.zeros_like(encoding)
                encoding_[:n] = encoding[:n]
                image = model.decode(encoding_).squeeze()

            image = data_module.unnormalize(image, im_format)
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            if i == 0:
                axis.set_ylabel(n, fontsize=16)
    plt.show()


@torch.no_grad()
def reconstruct_images(autoencoder, dataloader, normalized):
    torch.random.manual_seed(52)
    num_images = 8

    images, _ = next(iter(dataloader))
    fig, axes_mat = plt.subplots(ncols=num_images, nrows=2,
                                 gridspec_kw={'wspace': 0, 'hspace': 0})

    for i, axes in enumerate(axes_mat):
        for image, axis in zip(images, axes):
            # axis.axis('off')
            axis.set_xticks([])
            axis.set_yticks([])
            if i == 1:
                image = autoencoder(image.unsqueeze(0)).squeeze()
            if normalized:
                image = cifar10.unnormalize(image)
            else:
                image = image.permute(1, 2, 0)
            axis.imshow(image)
    plt.show()


@torch.no_grad()
def fcae_reconstruction_error_plot():
    # Reconstructing
    model_pickle = f'models/nestedDropoutAutoencoder_shallow_21-01-13__10-31-45_dict.pt'

    xticks = [1, 64, 128, 256, 512, 1024]
    dataloader = cifar10.get_dataloader(16)
    device = utils.get_device()
    autoencoder: Autoencoder = torch.load(open(model_pickle, 'rb'), map_location=device)['autoencoder']
    autoencoder.eval()

    repr_dim = autoencoder.repr_dim
    indices = [int(2 ** i) for i in torch.arange(math.log2(repr_dim) + 1)]
    reconst_loss = np.empty(len(indices))

    for i, index in tqdm(enumerate(indices), total=len(indices)):
        losses = []
        for sample, _ in dataloader:
            sample = sample.to(device)
            sample_rep = autoencoder.encode(sample)
            cut_repr = torch.zeros_like(sample_rep)
            cut_repr[:, :i + 1] = sample_rep[:, :i + 1]
            reconst = autoencoder.decode(cut_repr)
            _rec_loss.append(torch.linalg.norm(sample - reconst).item())
        reconstruction_loss.append(np.mean(_rec_loss))

    pickle.dump((reconst_loss, repr_dim),
                open(f'pickles/reconstruction_loss_{utils.current_time()}.pkl', 'wb'))

    # Plotting
    nd_reconst_loss, repr_dim = pickle.load(
        open('pickles/reconstruction_loss_21-01-14__13-10-00_nested_dropout.pkl', 'rb'))
    vanilla_reconst_loss, repr_dim = pickle.load(
        open('pickles/reconstruction_loss_21-01-14__13-24-42_vanilla.pkl', 'rb'))
    indices = [int(2 ** i) for i in torch.arange(math.log2(repr_dim) + 1)]

    plt.plot(indices, nd_reconst_loss, label='Nested dropout autoencoder')
    plt.plot(indices, vanilla_reconst_loss, label='Standard autoencoder')
    plt.xlabel('Representation Bytes')
    plt.ylabel('Reconstruction Error')
    plt.xticks(xticks)
    plt.title('Reconstruction Error by Representation Bits')
    plt.savefig('plots/reconstruction_error')
    plt.yscale('log')
    plt.legend()
    plt.show()


@torch.no_grad()
def get_reconstruction_error(autoencoder, dataloader, dim, device, subset=200):
    loss_func = nn.MSELoss()

    losses = [[] for _ in range(dim)]
    for i, (sample, _) in tqdm(enumerate(dataloader), total=subset):
        if i == subset:
            break

        sample = sample.to(device)
        encoding = autoencoder.encode(sample)
        for j in range(dim):
            cut_repr = torch.clone(encoding)
            cut_repr[:, j + 1:] = 0
            reconstruction = autoencoder.decode(cut_repr)
            losses[j].append(loss_func(reconstruction, sample).item())
    results = np.average(losses, axis=1)
    return results


@torch.no_grad()
def plot_cae_reconstruction_error(nd_autoencoder, dataloader):
    # Reconstructing
    repr_dim = 64
    device = utils.get_device()
    nd_autoencoder = nd_autoencoder.to(device)
    nested_dropout_losses = get_reconstruction_error(nd_autoencoder, dataloader, repr_dim, device)

    # Plotting
    indices = [int(2 ** i) for i in torch.arange(math.log2(repr_dim) + 1)]

    for i in range(2):
        plt.plot(range(1, repr_dim + 1), nested_dropout_losses)
        plt.xlabel('Representation Channels')
        plt.ylabel('Reconstruction Error')
        plt.xticks(indices)
        plt.title('Error by Channels')
        if i == 0:
            plt.yscale('log')
        plt.tight_layout()
        plt.show()


@torch.no_grad()
def plot_filters(model, title=None, output_shape=None):
    filter_matrix, _ = model.get_weights(1)
    shape = filter_matrix.shape
    channels = shape[1]

    if output_shape is None:
        output_shape = shape[:2]
    else:
        assert output_shape[1] % channels == 0
        filter_matrix = np.reshape(filter_matrix, (*output_shape, *shape[2:]))

    fig, axes_mat = plt.subplots(*output_shape, figsize=(output_shape[0] * 2, output_shape[1] * 2))
    for i, (filters, axes) in enumerate(zip(filter_matrix, axes_mat)):
        for j, (filter, axis) in enumerate(zip(filters, axes)):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.imshow(filter, cmap='Greys')
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def cae_plots(model_save: str, device, name=None):
    if name is not None:
        print(name)

    save_dict = torch.load(f'{model_save}/model.pt', map_location=device)
    nested_dropout = 'p' in save_dict

    if os.path.basename(model_save).startswith('cae'):
        model = ConvAutoencoder(**save_dict)
        if nested_dropout:
            model = NestedDropoutAutoencoder(model, **save_dict)
        title = 'Convolutional Autoencoder'

    elif os.path.basename(model_save).startswith('classifier'):
        model = Classifier(**save_dict)
        title = 'Classifier'

    else:
        raise NotImplementedError()

    model.eval()
    try:
        model.load_state_dict(save_dict['model'])
    except:
        print('Save dict mismatch\n')
        return None

    print(utils.get_num_parameters(model), '\n')
    if nested_dropout:
        title = f'Nested Dropout {title}'
    if name is not None:
        title += '\n' + name
    plot_filters(model, title=title, output_shape=(8, 4))

    # if nested_dropout:
    #     plot_images_by_channels(model, imagenette, True, 'Y')
    #     dataloader = imagenette.get_dataloader(normalize=True, image_mode='Y')
    #     plot_cae_reconstruction_error(model, dataloader)


def main():
    device = utils.get_device()
    model_saves = ['saves/cae-F-ConvAutoencoder_21-05-06--09-26-54',
                   'saves/cae-F-NestedDropoutAutoencoder_21-05-06--00-27-18']
    # model_saves = ['saves/cae-F-ConvAutoencoder_21-05-05--13-56-35',
    #                'saves/cae-F-NestedDropoutAutoencoder_21-05-05--15-18-07']
    # for model_save in model_saves:
    #     cae_plots(model_save, device)
    for model_save in os.listdir('saves'):
        if model_save.startswith('cae'):
            continue
        cae_plots('saves/' + model_save, device, name=model_save)


if __name__ == '__main__':
    main()
