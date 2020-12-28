import torch


def get_data_representation(autoencoder, dataloader):
    with torch.no_grad():
        reps = torch.stack(
            [autoencoder.get_representation(batch) for batch, _ in dataloader]).view(-1, autoencoder.repr_dim)
    return reps


def restore_image(data, mean, std) -> torch.tensor:
    restored_image = torch.tensor(std).view(3, 1, 1) * data + torch.tensor(mean).view(3, 1, 1)
    return restored_image.permute(1, 2, 0)
