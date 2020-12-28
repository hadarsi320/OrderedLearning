import torch


def get_data_representation(autoencoder, dataset):
    with torch.no_grad():
        reps = torch.stack(
            [autoencoder.get_repr(x) for x, _ in dataset]).squeeze()
    return reps
