import torch.nn as nn


class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.bn(self.conv1(x))))
