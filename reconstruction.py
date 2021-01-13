import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from nueral_networks.autoencoders import Autoencoder
from utils_package import data_utils, utils


def main():
    model_pickle = f'models/nestedDropoutAutoencoder_shallow_ReLU_21-01-13__02-50-53_dict.pt'

    dataloader = data_utils.get_cifar10_dataloader(1000)
    device = utils.get_device()
    autoencoder: Autoencoder = torch.load(open(model_pickle, 'rb'), map_location=device)['autoencoder']
    autoencoder.eval()

    repr_dim = autoencoder.repr_dim
    reconst_loss = np.empty(repr_dim)

    for i in tqdm(range(repr_dim)):
        losses = []
        for sample, _ in dataloader:
            sample = sample.to(device)
            sample_repr = autoencoder.encode(sample)
            sample_repr[:, i + 1:] = 0
            reconst = autoencoder.decode(sample_repr)
            losses.append(torch.mean((sample - reconst) ** 2).item())
        reconst_loss[i] = np.mean(losses)

    pickle.dump((reconst_loss, repr_dim),
                open(f'pickles/reconstruction_loss_{utils.current_time()}.pkl', 'wb'))

    plt.plot(range(1, repr_dim + 1), reconst_loss)
    plt.xlabel('Representation Bytes')
    plt.ylabel('Reconstruction Error')
    plt.xticks(list(range(0, repr_dim, 100))[:-1] + [repr_dim])
    plt.title('Reconstruction Error by Representation Bytes')
    plt.savefig('plots/reconstruction_error')
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    main()
