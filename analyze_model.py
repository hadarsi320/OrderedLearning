import torch

from utils_package import utils, data_utils


def main():
    autoencoder = torch.load(open('models/nestedDropoutAutoencoder_shallow_100_ReLU_21_01_04_14_19_00.pkl', 'rb'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    _, train_loader = data_utils.load_cifar10()
    utils.plot_repr_var(autoencoder, train_loader, device, show=True, title='Final Representation Variance')


if __name__ == '__main__':
    main()
