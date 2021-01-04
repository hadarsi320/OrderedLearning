import torch
import matplotlib.pyplot as plt


def get_data_representation(autoencoder, dataloader, device):
    with torch.no_grad():
        reps = torch.stack(
            [autoencoder.get_representation(batch.to(device)) for batch, _ in dataloader])\
            .view(-1, autoencoder.repr_dim)
    return reps


def restore_image(data, mean, std) -> torch.tensor:
    restored_image = torch.tensor(std).view(3, 1, 1) * data + torch.tensor(mean).view(3, 1, 1)
    return restored_image.permute(1, 2, 0)


def binarize_data(data: torch.Tensor, q=0.5):
    if not 0 < q < 1:
        raise ValueError('q must be in range (0, 1)')

    binarized = torch.zeros_like(data)
    q_quantiles = torch.quantile(data, q, dim=0)
    binarized[data >= q_quantiles] = 1
    return binarized


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
