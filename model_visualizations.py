import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import utils
from data import cifar10
from models.autoencoders import *
from torch.linalg import norm


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
def compare_num_channels(model_dir):
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
def get_reconstruction_error(autoencoder, dataloader, dim, device):
    loss_func = nn.MSELoss()

    losses = [[] for _ in range(dim)]
    for i, (sample, _) in tqdm(enumerate(dataloader)):
        if i == 100:
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
def cae_reconstruction_error_plot(nd_autoencoder, dataloader):
    # Reconstructing
    repr_dim = 32
    device = utils.get_device()
    nd_autoencoder = nd_autoencoder.to(device)
    nested_dropout_losses = get_reconstruction_error(nd_autoencoder, dataloader, repr_dim, device)

    # Plotting
    indices = [int(2 ** i) for i in torch.arange(math.log2(repr_dim) + 1)]

    plt.plot(range(1, repr_dim + 1), nested_dropout_losses)
    plt.xlabel('Representation Bits')
    plt.ylabel('Reconstruction Error')
    plt.xticks(indices)
    plt.title('Reconstruction Error by Channels')
    # plt.savefig('plots/reconstruction_error')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


def plot_filters(cae):
    filter_matrix = cae.get_weights(1)[:3]
    fig, axes_mat = plt.subplots(nrows=filter_matrix.shape[0], ncols=filter_matrix.shape[1])
    for filters, axes in zip(filter_matrix, axes_mat):
        for filter, axis in zip(filters, axes):
            axis.axis('off')
            axis.imshow(filter)
    plt.show()


def main():
    vl_model_dir = '/mnt/ml-srv1/home/hadarsi/ordered_learning/saves/*cae-C-ConvAutoencoder_21-04-20--16-43-34/'
    ne_model_dir = '/mnt/ml-srv1/home/hadarsi/ordered_learning/saves/*cae-C-NestedDropoutAutoencoder_21-04-20--16-55' \
                   '-48/'

    save_dict = torch.load(f'{vl_model_dir}/model.pt')
    normalized = save_dict['normalize_data']
    vl_autoencoder = ConvAutoencoder(**save_dict)
    vl_autoencoder.load_state_dict(save_dict['model'])
    vl_autoencoder.eval()

    save_dict = torch.load(f'{ne_model_dir}/model.pt')
    assert normalized == save_dict['normalize_data']
    nd_autoencoder = NestedDropoutAutoencoder(ConvAutoencoder(**save_dict), **save_dict)
    nd_autoencoder.load_state_dict(save_dict['model'])
    nd_autoencoder.eval()
    dataloader = cifar10.get_dataloader(128, normalize=normalized)

    # compare_num_channels(ne_model_dir)
    # cae_reconstruction_error_plot(nd_autoencoder, dataloader)
    # utils.plot_repr_var(nd_autoencoder, dataloader, show=True, ylabel='Channels')
    plot_filters(nd_autoencoder)


if __name__ == '__main__':
    main()
