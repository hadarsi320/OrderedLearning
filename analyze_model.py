import os

import torch
import torch.nn.functional as F

import model_visualizations
import utils
from data import cifar10


def main():
    device = utils.get_device()
    dataloader = cifar10.get_dataloader(batch_size=1000)

    losses_dict = {}
    for file in os.listdir('models/'):
        if file.endswith('dict.pt'):
            model_dict = torch.load(f'models/{file}', map_location=device)
            autoencoder = model_dict['autoencoder']
            model_params = model_dict['parameters']

            print(f'{file}:')
            for key in model_params:
                print(f'\t{key}: {model_params[key]}')

            if 'converged_unit' in model_params:
                print(f'\tconverged unit: {model_dict["converged_unit"]}')
            loss = utils.get_model_loss(autoencoder, dataloader, lambda x, y, res: F.mse_loss(x, res), device)
            print(f'\tmodel loss: {loss:.2f}')
            print()

            model_visualizations.plot_repr_var(autoencoder, dataloader, device, title=file, show=True)
            losses_dict[file] = loss

    best_models = sorted(losses_dict, key=lambda x: losses_dict[x])[:5]
    for file in best_models:
        print(file, losses_dict[file])
        # autoencoder = torch.load(f'models/{file}', map_location=device)['autoencoder']
        # utils.plot_repr_var(autoencoder, dataloader, device, title=file, show=True)


if __name__ == '__main__':
    main()
