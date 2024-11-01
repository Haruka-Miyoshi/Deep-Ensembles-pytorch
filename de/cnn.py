import torch
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self, in_channels:int=3, n_class:int=10):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.n_class = n_class

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc(x)
        return x