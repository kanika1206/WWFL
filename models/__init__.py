import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """LeNet for MNIST (1x28x28 input)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)) + x)
        return out


class ResNet9(nn.Module):
    """ResNet9 for CIFAR10 (3x32x32 input)."""
    def __init__(self, num_classes=10):
        super().__init__()
        def conv_bn(ic, oc, ks=3, stride=1, pad=1):
            return nn.Sequential(
                nn.Conv2d(ic, oc, ks, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True)
            )
        self.prep   = conv_bn(3, 64)
        self.layer1 = nn.Sequential(conv_bn(64, 128), nn.MaxPool2d(2))
        self.res1   = ResidualBlock(128, 128)
        self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2))
        self.res2   = ResidualBlock(512, 512)
        self.pool   = nn.AdaptiveMaxPool2d(1)
        self.fc     = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.res1(self.layer1(x))
        x = self.layer2(x)
        x = self.res2(self.layer3(x))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
