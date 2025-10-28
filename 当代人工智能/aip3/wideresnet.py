import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)  # Dropout after the first convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)  # Dropout after the second convolution

        # If the number of input channels is not equal to the number of output channels,
        # add a 1x1 convolution to match the dimensions
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.dropout1(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout2(self.bn2(self.conv2(out)))

        # Shortcut connection
        shortcut = self.shortcut(residual)

        out += shortcut
        out = self.relu(out)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=4, num_classes=10, dropout_rate=0.3):
        super(WideResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        n = (depth - 4) // 6
        k = widen_factor

        self.group1 = self._make_group(ResidualBlock, 16, n, 1, k, dropout_rate)
        self.group2 = self._make_group(ResidualBlock, 32, n, 2, k, dropout_rate)
        self.group3 = self._make_group(ResidualBlock, 64, n, 2, k, dropout_rate)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_group(self, block, out_channels, num_blocks, stride, k, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels * k, stride, dropout_rate))
            self.in_channels = out_channels * k
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



