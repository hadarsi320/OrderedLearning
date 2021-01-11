import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from nueral_networks.autoencoders import Autoencoder
from utils_package import data_utils


def main():
    model_pickle = f'models/nestedDropoutAutoencoder_deep_ReLU_21-01-07__01-18-13.pkl'

    dataset, dataloader = data_utils.get_cifar10_dataloader(1000)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    autoencoder: Autoencoder = torch.load(open(model_pickle, 'rb'), map_location=device)
    autoencoder.eval()

    repr_dim = autoencoder.repr_dim
    reconstruction_loss = []

    for i in tqdm(range(repr_dim)):
        _rec_loss = []
        for sample, _ in dataloader:
            sample = sample.to(device)
            repr = autoencoder.get_representation(sample)
            cut_repr = torch.zeros_like(repr)
            cut_repr[:, :i+1] = repr[:, :i+1]
            reconst = autoencoder.get_reconstructions(cut_repr)
            _rec_loss.append(torch.linalg.norm(sample - reconst))
        reconstruction_loss.append(torch.mean(torch.tensor(_rec_loss)))

    plt.xlabel('Representation Bytes')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error by Representation Bytes')
    plt.plot(range(1, repr_dim + 1), reconstruction_loss)
    plt.savefig('Reconstruction Error')


if __name__ == '__main__':
    main()
