import torch
from torch import nn


class CIFARModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_maxp = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2)
        )
        self.flat = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_maxp(x)
        x = self.flat(x)
        x = self.mlp(x)
        return x


if __name__ == '__main__':
    input = torch.zeros((64, 3, 32, 32))  # N, C, W, H
    model = CIFARModel()
    output = model(input)
    print(output.shape)