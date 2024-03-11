import torch
from torch import nn


class MyConv(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x + 1
        return x


mypic = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])  # (5, 5)

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])  # (3, 3)

mypic = torch.reshape(mypic, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
