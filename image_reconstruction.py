import matplotlib.pyplot as plt
import torch

from nueral_networks.autoencoders import Autoencoder
from utils_package import cifar_utils, utils


@torch.no_grad()
def main():
    model_pickle = 'models/nestedDropoutAutoencoder_shallow_ReLU_21-01-13__02-50-53_dict.pt'
    torch.random.manual_seed(42)
    num_images = 6
    code_lengths = [16, 64, 128, 256, 1024, 'Original']

    device = utils.get_device()
    model: Autoencoder = torch.load(model_pickle, map_location=device)['autoencoder']
    model.eval()

    dataset = cifar_utils.get_cifar10_dataloader().dataset

    def restore(im: torch.Tensor):
        return utils.restore_image(im.cpu().view(3, 32, 32), cifar_utils.CIFAR10_MEAN, cifar_utils.CIFAR10_STD)

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
            axis[i].imshow(restore(image))
            axis[i].set_xticks([])
            axis[i].set_yticks([])
            if i == 0:
                axis[i].set_ylabel(code_length, fontsize=16)

    plt.tick_params()

    plt.show()


if __name__ == '__main__':
    main()
