from typing import Tuple, Optional

import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, input_size: int, output_size: int, layers: Tuple[int, ...], head: Optional):
        super().__init__()
        if not layers:
            raise ValueError("There should be at least 1 hidden layer")
        self.model = nn.ModuleList([nn.Linear(input_size, layers[0])])
        self.model.extend([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.model.append(nn.Linear(layers[-1], output_size))
        self.head = head

    def forward(self, x):
        for layer in self.model[:-1]:
            x = torch.relu(layer(x))
        if self.head:
            return self.head(self.model[-1](x))
        else:
            return self.model[-1](x)

    def save(self, path):
        self