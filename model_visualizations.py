import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

import utils
import utils.image_utils
from data import imagenette
from models import ConvAutoencoder


# @torch.no_grad()
# def compare_code_lengths():
#     model_pickle = 'models/nestedDropoutAutoencoder_shallow_ReLU_21-01-13__02-50-53_dict.pt'
#     torch.random.manual_seed(42)
#     num_images = 6
#     code_lengths = [16, 64, 128, 256, 1024, 'Original']
#
#     device = utils.get_device()
#     model: Autoencoder = torch.load(model_pickle, map_location=device)['autoencoder']
#     model.eval()
#
#     dataset = cifar10.get_dataloader().dataset
#
#     plt.tight_layout()
#     fig, axes = plt.subplots(ncols=num_images, nrows=len(code_lengths), squeeze=False, figsize=(12, 12))
#     indices = torch.randint(len(dataset), (num_images,))
#     for i, index in enumerate(indices):
#         original_image, _ = dataset[index]
#         encoding = model.encode(original_image.to(device))
#         for code_length, axis in zip(code_lengths, axes):
#             if code_length == 'Original':
#                 image = original_image
#             else:
#                 if code_length == 'Full':
#                     encoding_ = encoding
#                 else:
#                     encoding_ = torch.zeros_like(encoding)
#                     encoding_[:code_length] = encoding[:code_length]
#                 image = model.decode(encoding_)
#             axis[i].imshow(cifar10.unnormalize(image))
#             axis[i].set_xticks([])
#             axis[i].set_yticks([])
#             if i == 0:
#                 axis[i].set_ylabel(code_length, fontsize=16)
#
#     plt.show()
#
#
# @torch.no_grad()
# def plot_repr_var(autoencoder, dataloader, scale='log', show=False, **kwargs):
#     device = kwargs.get('device', get_device())
#     autoencoder = autoencoder.to(device)
#
#     plt.clf()
#     reprs = get_data_representation(autoencoder, dataloader, device)
#     if reprs.dim() == 4:
#         plt.plot(torch.mean(torch.var(reprs, dim=[0]), dim=[1, 2]).cpu())
#     else:
#         plt.plot(torch.var(reprs, dim=0).cpu())
#     plt.yscale(scale)
#     plt.xlabel(kwargs.get('xlabel', 'Units'))
#     plt.ylabel(kwargs.get('ylabel', 'Variance'))
#
#     if 'title' in kwargs:
#         plt.title(kwargs.pop('title'))
#
#     if 'savefig' in kwargs:
#         plt.savefig(kwargs.pop('savefig'))
#
#     if show:
#         plt.show()
#
#
# @torch.no_grad()
# def fcae_reconstruction_error_plot():
#     # Reconstructing
#     model_pickle = f'models/nestedDropoutAutoencoder_shallow_21-01-13__10-31-45_dict.pt'
#
#     xticks = [1, 64, 128, 256, 512, 1024]
#     dataloader = cifar10.get_dataloader(16)
#     device = utils.get_device()
#     autoencoder: Autoencoder = torch.load(open(model_pickle, 'rb'), map_location=device)['autoencoder']
#     autoencoder.eval()
#
#     repr_dim = autoencoder.repr_dim
#     indices = [int(2 ** i) for i in torch.arange(math.log2(repr_dim) + 1)]
#     reconst_loss = np.empty(len(indices))
#
#     for i, index in tqdm(enumerate(indices), total=len(indices)):
#         losses = []
#         for sample, _ in dataloader:
#             sample = sample.to(device)
#             sample_rep = autoencoder.encode(sample)
#             cut_repr = torch.zeros_like(sample_rep)
#             cut_repr[:, :i + 1] = sample_rep[:, :i + 1]
#             reconst = autoencoder.decode(cut_repr)
#             _rec_loss.append(torch.linalg.norm(sample - reconst).item())
#         reconstruction_loss.append(np.mean(_rec_loss))
#
#     pickle.dump((reconst_loss, repr_dim),
#                 open(f'pickles/reconstruction_loss_{utils.current_time()}.pkl', 'wb'))
#
#     # Plotting
#     nd_reconst_loss, repr_dim = pickle.load(
#         open('pickles/reconstruction_loss_21-01-14__13-10-00_nested_dropout.pkl', 'rb'))
#     vanilla_reconst_loss, repr_dim = pickle.load(
#         open('pickles/reconstruction_loss_21-01-14__13-24-42_vanilla.pkl', 'rb'))
#     indices = [int(2 ** i) for i in torch.arange(math.log2(repr_dim) + 1)]
#
#     plt.plot(indices, nd_reconst_loss, label='Nested dropout autoencoder')
#     plt.plot(indices, vanilla_reconst_loss, label='Standard autoencoder')
#     plt.xlabel('Representation Bytes')
#     plt.ylabel('Reconstruction Error')
#     plt.xticks(xticks)
#     plt.title('Reconstruction Error by Representation Bits')
#     plt.savefig('plots/reconstruction_error')
#     plt.yscale('log')
#     plt.legend()
#     plt.show()


@torch.no_grad()
def plot_images_by_channels(model: ConvAutoencoder, data_module, normalized, im_format='RGB', show=True):
    torch.random.manual_seed(42)
    num_channels = [1, 2, 4, 8, 16, 32, 64, 'Original']
    num_images = len(num_channels)
    model.to(torch.device('cpu'))

    images, _ = next(iter(data_module.get_dataloader(batch_size=num_images, normalize=normalized)))

    plt.tight_layout()
    fig, axes_mat = plt.subplots(ncols=num_images, nrows=len(num_channels), squeeze=False,
                                 figsize=(num_images * 2, num_images * 2))
    for i, (original_image, axes) in enumerate(zip(images, axes_mat.transpose())):
        # encoding = model.encode(original_image).squeeze()
        encoding = model.get_feature_map(original_image, 1).squeeze()
        for n, axis in zip(num_channels, axes):
            if n == 'Original':
                image = original_image
            else:
                encoding_ = torch.zeros_like(encoding)
                encoding_[:n] = encoding[:n]
                # image = model.decode(encoding_).squeeze()
                image = model.forward_feature_map(encoding_, 1).squeeze()

            image = data_module.unnormalize(image, im_format)
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            if i == 0:
                axis.set_ylabel(n, fontsize=16)
    if show:
        plt.show()


@torch.no_grad()
def reconstruct_images(autoencoder, dataloader, normalized, data_module):
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
                image = data_module.unnormalize(image)
            else:
                image = image.permute(1, 2, 0)
            axis.imshow(image)
    plt.show()


@torch.no_grad()
def get_reconstruction_error(autoencoder: ConvAutoencoder, dataloader, dim, device, frac=0.1):
    loss_func = nn.MSELoss()
    subset = round(len(dataloader) * frac)

    losses = [[] for _ in range(dim)]
    for i, (sample, _) in tqdm(enumerate(dataloader), total=subset):
        if i == subset:
            break

        sample = sample.to(device)
        # encoding = autoencoder.encode(sample)
        encoding = autoencoder.get_feature_map(sample, 1)
        for j in range(dim):
            cut_repr = torch.clone(encoding)
            cut_repr[:, j + 1:] = 0
            # reconstruction = autoencoder.decode(cut_repr)
            reconstruction = autoencoder.forward_feature_map(cut_repr, 1)
            losses[j].append(loss_func(reconstruction, sample).item())
    results = np.average(losses, axis=1)
    return results


@torch.no_grad()
def plot_conv_autoencoder_reconstruction_error(autoencoder, dataloader, repr_dim=64, show=True, scale='linear'):
    # Reconstructing
    device = utils.get_device()
    autoencoder = autoencoder.to(device)
    nested_dropout_losses = get_reconstruction_error(autoencoder, dataloader, repr_dim, device)

    # Plotting
    indices = [int(2 ** i) for i in torch.arange(math.log2(repr_dim) + 1)]

    for i in range(2):
        plt.plot(range(1, repr_dim + 1), nested_dropout_losses)
        plt.xlabel('Representation Channels')
        plt.ylabel('Reconstruction Error')
        plt.xticks(indices)
        plt.title('Error by Channels')
        plt.yscale(scale)
        plt.tight_layout()
        if show:
            plt.show()


@torch.no_grad()
def plot_filters(model, title=None, output_shape=None, cmap='Greys', show=True, normalize=False):
    filter_matrix = model.get_weights(0)[0].cpu().numpy()
    shape = filter_matrix.shape
    channels = shape[1]

    if output_shape is None:
        output_shape = utils.square_shape(*shape[:2])
    else:
        assert output_shape[1] % channels == 0
        assert shape[0] * shape[1] == output_shape[0] * output_shape[1]
    filter_matrix = np.reshape(filter_matrix, (*output_shape, *shape[2:]))

    kwargs = {}
    if normalize:
        kwargs['vmin'] = filter_matrix.min()
        kwargs['vmax'] = filter_matrix.max()
    utils.plot_subfigures(filter_matrix, cmap=cmap, title=title, show=show, **kwargs)


@torch.no_grad()
def plot_feature_maps(conv_layer, image):
    # plt.imshow(image)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Original Image')
    # plt.show()

    feature_maps = conv_layer(image).numpy()
    shape = feature_maps.shape
    shape[:2] = utils.square_shape(*shape[:2])
    np.reshape(feature_maps, shape)

    utils.plot_subfigures(feature_maps, title='Feature Maps')


@torch.no_grad()
def model_plots(model_save: str, device, name=None, image=None, filters_normalize=False):
    if name is not None:
        print(name)

    save_dict, model = utils.load_model(model_save, device)
    if save_dict is None:
        return None

    if 'p' in save_dict or ('nested_dropout' in save_dict and save_dict['nested_dropout']['apply_nested_dropout']):
        apply_nested_dropout = True
    else:
        apply_nested_dropout = False

    if os.path.basename(model_save).startswith('cae') or \
            os.path.basename(model_save).startswith('ConvAutoencoder'):
        title = 'Convolutional Autoencoder'
    elif os.path.basename(model_save).startswith('classifier'):
        title = 'Classifier'

    model.eval()
    print(f'The model has {utils.get_num_parameters(model)} parameters', '\n')

    if apply_nested_dropout:
        title = f'Nested Dropout {title}'
    if name is not None:
        title += '\n' + name
    plot_filters(model, title=title, normalize=filters_normalize)

    # if image is not None:
    #     conv = list(list(model.children())[-1].children())[0]
    #     plot_feature_maps(conv, image)

    # if apply_nested_dropout:
    #     plot_images_by_channels(model, imagenette, True, 'Y')
    #     dataloader = imagenette.get_dataloader(normalize=True, image_mode='Y')
    #     plot_cae_reconstruction_error(model, dataloader)


def compare_models(vanilla_dir, dropout_dir, device):
    _, vanilla_model = utils.load_model(vanilla_dir, device)
    _, dropout_model = utils.load_model(dropout_dir, device)
    plot_images_by_channels(vanilla_model, imagenette, True, 'Y')
    plot_images_by_channels(dropout_model, imagenette, True, 'Y')

    dataloader = imagenette.get_dataloader(normalize=True, image_mode='Y')

    repr_dim = 64
    vanilla_model.to(device)
    vanilla_losses = get_reconstruction_error(vanilla_model, dataloader, repr_dim, device)
    dropout_model.to(device)
    dropout_losses = get_reconstruction_error(dropout_model, dataloader, repr_dim, device)
    # Plotting
    indices = [int(2 ** i) for i in torch.arange(math.log2(repr_dim) + 1)]

    for i in range(2):
        plt.plot(range(1, repr_dim + 1), vanilla_losses, label='Vanilla Model')
        plt.plot(range(1, repr_dim + 1), dropout_losses, label='Nested Dropout Model')
        plt.xlabel('Representation Channels')
        plt.ylabel('Reconstruction Error')
        plt.xticks(indices)
        plt.title('Error by Channels')
        if i == 0:
            plt.yscale('log')
        plt.tight_layout()
        plt.legend()
        plt.show()


def main():
    device = utils.get_device()
    saves_dir = 'saves'
    # compare_models(f'{saves_dir}/cae-F-ConvAutoencoder_21-06-01--10-43-39',
    #                f'{saves_dir}/cae-F-ConvAutoencoder_21-06-03--08-16-48',
    #                device)

    saves = ['ConvAutoencoder-F_21-06-09--04-57-08',
             'ConvAutoencoder-F_21-06-09--12-32-29',
             'ConvAutoencoder-F_21-06-16--19-26-45']
    for save in saves:
        model_plots(saves_dir + '/' + save, device, filters_normalize=False, name=save)
        model_plots(saves_dir + '/' + save, device, filters_normalize=True, name=save)


if __name__ == '__main__':
    main()
