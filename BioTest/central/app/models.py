import torch.nn as nn
import torch.nn.functional as F
import torch


class InputEncoder(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        ...


class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin2 = nn.Linear(32, 32)

    def forward(self, x):
        x = F.relu(self.lin2(x))
        return x


class OutputDecoder(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        ...
