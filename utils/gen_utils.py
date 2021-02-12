from datetime import datetime

import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def get_data_representation(autoencoder, dataloader, device):
    return torch.cat([autoencoder.get_representation(batch.to(device)) for batch, _ in dataloader])


def restore_image(data, mean, std) -> torch.tensor:
    restored_image = torch.tensor(std).view(3, 1, 1) * data + torch.tensor(mean).view(3, 1, 1)
    return restored_image.permute(1, 2, 0)


def binarize_data(data: torch.Tensor, bin_quantile=0.5):
    if not 0 < bin_quantile < 1:
        raise ValueError('The binarization quantile must be in range (0, 1)')

    binarized = torch.zeros_like(data)
    q_quantiles = torch.quantile(data, bin_quantile, dim=0)
    binarized[data >= q_quantiles] = 1
    return binarized


@torch.no_grad()
def plot_repr_var(autoencoder, train_loader, device,
                  show=False, **kwargs):
    plt.clf()
    reprs = get_data_representation(autoencoder, train_loader, device)
    plt.plot(torch.var(reprs, dim=0).to('cpu'))

    if 'title' in kwargs:
        plt.title(kwargs.pop('title'))

    if 'savefig' in kwargs:
        plt.savefig(kwargs.pop('savefig'))

    if show:
        plt.show()


def current_time():
    return datetime.now().strftime('%y-%m-%d__%H-%M-%S')


def get_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() == 8:
            return torch.device('cuda:7')
        return torch.device('cuda')
    return torch.device('cpu')


# TODO create a function which saves a model in a dir with a log file
