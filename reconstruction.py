import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import utils
from data import cifar10
from models.autoencoder import Autoencoder


def main():
    # Reconstructing
    model_pickle = f'models/nestedDropoutAutoencoder_shallow_21-01-13__10-31-45_dict.pt'

    dataloader = cifar10.get_dataloader(1000)
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
    plt.xticks([1, 64, 128, 256, 512, 1024])
    plt.title('Reconstruction Error by Representation Bits')
    plt.legend()
    plt.savefig('plots/reconstruction_error')
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    main()
