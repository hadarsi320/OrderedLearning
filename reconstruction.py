import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from models.autoencoder import Autoencoder
from utils import gen_utils
from data import cifar10


def main():
    model_pickle = f'models/nestedDropoutAutoencoder_shallow_ReLU_21-01-12__04-17-04_dict.pt'

    dataloader = cifar10.get_cifar10_dataloader(1000)
    device = gen_utils.get_device()
    autoencoder: Autoencoder = torch.load(open(model_pickle, 'rb'), map_location=device)['autoencoder']
    autoencoder.eval()

    repr_dim = autoencoder.repr_dim
    reconstruction_loss = []

    for i in tqdm(range(repr_dim)):
        _rec_loss = []
        for sample, _ in dataloader:
            sample = sample.to(device)
            sample_rep = autoencoder.encode(sample)
            cut_repr = torch.zeros_like(sample_rep)
            cut_repr[:, :i + 1] = sample_rep[:, :i + 1]
            reconst = autoencoder.decode(cut_repr)
            _rec_loss.append(torch.linalg.norm(sample - reconst).item())
        reconstruction_loss.append(np.mean(_rec_loss))

    pickle.dump((reconstruction_loss, repr_dim),
                open(f'pickles/reconstruction_loss_{gen_utils.current_time()}.pkl', 'wb'))

    plt.plot(range(1, repr_dim + 1), reconstruction_loss)
    plt.xlabel('Representation Bytes')
    plt.ylabel('Reconstruction Error')
    plt.xticks(list(range(0, repr_dim, 100)) + [repr_dim])
    plt.title('Reconstruction Error by Representation Bytes')
    plt.savefig('plots/reconstruction_error')
    plt.show()


if __name__ == '__main__':
    main()
