from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn


def mish(tensor):
    return tensor * torch.tanh(F.softplus(tensor))


class Mish(nn.Module):
    def __init__(self): super().__init__()

    def forward(self, tensor): return mish(tensor)


class MLP(nn.Module):

    def __init__(self, input_size: int, output_size: int, layers: Tuple[int, ...], head: Optional):
        super().__init__()
        if not layers:
            raise ValueError("There should be at least 1 hidden layer")
        self.model = nn.ModuleList([nn.Linear(input_size, layers[0])])
        for i in range(len(layers) - 1):
            self.model.extend([nn.Linear(layers[i], layers[i + 1]), nn.ReLU()])
        self.model.append(nn.Linear(layers[-1], output_size))
        if head:
            self.model.append(head)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
