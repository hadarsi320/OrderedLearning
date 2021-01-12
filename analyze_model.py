import os

import torch
import torch.nn.functional as F

from utils_package import utils, data_utils, nn_utils


def main():
    device = utils.get_device()
    dataloader = data_utils.get_cifar10_dataloader(1000)

    for file in os.listdir('models/'):
        if file.endswith('pt'):
            model_dict = torch.load(f'models/{file}', map_location=device)
            autoencoder = model_dict['autoencoder']
            converged_unit = model_dict['converged_unit']
            model_params = model_dict['parameters']
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
