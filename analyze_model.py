import os

import torch
import torch.nn.functional as F

from utils_package import utils, nn_utils
from data import cifar10


def main():
    device = utils.get_device()
    dataloader = cifar10.get_cifar10_dataloader(1000)

    for file in os.listdir('models/'):
        if file.endswith('pt'):
            dict = torch.load(f'models/{file}', map_location=device)
            autoencoder = dict['autoencoder']
            converged_unit = dict['converged_unit']
            model_params = dict['parameters']
            print(f'{file}:')
            for key in model_params:
                print(f'\t{key}: {model_params[key]}')
            print(f'\tconverged unit: {converged_unit}')
            loss = nn_utils.get_model_loss(autoencoder, dataloader, lambda x, y, res: F.mse_loss(x, res), device)
            print(f'\tmodel loss: {loss:.2f}')
            print()

            utils.plot_repr_var(autoencoder, dataloader, device, title=file, show=True)


if __name__ == '__main__':
    main()
