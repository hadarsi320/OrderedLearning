import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_cifar10(batch_size):
    mean = [0.49139968, 0.48215841, 0.44653091]
    std = [0.24703223, 0.24348513, 0.26158784]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std),
                                    transforms.Lambda(lambda t: t.view(-1))])
    trainset = torchvision.datasets.CIFAR10(root='data/',
                                            transform=transform, train=True)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainset, train_loader
