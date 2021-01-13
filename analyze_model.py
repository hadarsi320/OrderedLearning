import os

import torch
import torch.nn.functional as F

from utils_package import utils, data_utils, nn_utils


def main():
    device = utils.get_device()
    dataloader = data_utils.get_cifar10_dataloader(1000)

    accuracies_dict = {}
    for file in os.listdir('models/'):
        if file.endswith('dict.pt'):
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

            accuracies_dict[file] = loss

    best_models = sorted(accuracies_dict, key=lambda x: accuracies_dict[x])[:5]
    for file in best_models:
        autoencoder = torch.load(f'models/{file}', map_location=device)['autoencoder']
        utils.plot_repr_var(autoencoder, dataloader, device, title=file, show=True)


if __name__ == '__main__':
    main()
