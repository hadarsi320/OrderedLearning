import torch

from nueral_networks.autoencoders import Autoencoder
from utils_package import data_utils, utils

import matplotlib.pyplot as plt


@torch.no_grad()
def main():
    model_pickle = 'models/nestedDropoutAutoencoder_shallow_ReLU_21-01-13__02-50-53_dict.pt'
    torch.random.manual_seed(42)
    num_images = 6
    code_lengths = [16, 64, 128, 256, -1, None]

    device = utils.get_device()
    model: Autoencoder = torch.load(model_pickle, map_location=device)['autoencoder']
    model.eval()

    dataset = data_utils.get_cifar10_dataloader().dataset

    def restore(im: torch.Tensor):
        return utils.restore_image(im.cpu().view(3, 32, 32), data_utils.CIFAR10_MEAN, data_utils.CIFAR10_STD)

    fig, axes = plt.subplots(ncols=num_images, nrows=len(code_lengths), squeeze=False, figsize=(12, 12))

    indices = torch.randint(len(dataset), (num_images,))
    for i, index in enumerate(indices):
        original_image, _ = dataset[index]
        encoding = model.encode(original_image.to(device))
        for code_length, axis in zip(code_lengths, axes):
            if code_length is not None:
                if code_length == -1:
                    encoding_ = encoding
                else:
                    encoding_ = torch.zeros_like(encoding)
                    encoding_[:code_length] = encoding[:code_length]
                image = model.decode(encoding_)
            else:
                image = original_image
            axis[i].imshow(restore(image))
    plt.tick_params()
    plt.show()


if __name__ == '__main__':
    main()
