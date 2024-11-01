import torch
from torch import nn
from torch.nn import functional as F

from .cnn import CNN

class Model(nn.Module):
    def __init__(self, in_channels:int, n_class:int, n_sample:int):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.n_class = n_class
        self.n_sample = n_sample

        self.experts = nn.ModuleList( [ CNN(in_channels=self.in_channels, n_class=self.n_class) for _ in range(self.n_sample) ] )

    def forward(self, x):
        sample_y = [ experts(x) for experts in self.experts ]
        return torch.stack(sample_y, dim=1)