import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

train_data = torchvision.datasets.CIFAR10('CIFAR', train=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('CIFAR', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)


class CIFARModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
