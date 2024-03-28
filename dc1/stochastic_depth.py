import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, prob_skip=0.5):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        self.prob_skip = prob_skip
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.training and torch.rand(1) < self.prob_skip:
            out += self.shortcut(residual)

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, prob_skip=0.5):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1, prob_skip=prob_skip)
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2, prob_skip=prob_skip)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2, prob_skip=prob_skip)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride, prob_skip):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, prob_skip))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, prob_skip=prob_skip))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(prob_skip=0.5, num_classes=10):
    return ResNet(ResidualBlock, [2, 2, 2], num_classes=num_classes, prob_skip=prob_skip)


def ResNet34(prob_skip=0.5, num_classes=10):
    return ResNet(ResidualBlock, [3, 4, 6], num_classes=num_classes, prob_skip=prob_skip)


def ResNet50(prob_skip=0.5, num_classes=10):
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=num_classes, prob_skip=prob_skip)


def ResNet101(prob_skip=0.5, num_classes=10):
    return ResNet(ResidualBlock, [3, 4, 23, 3], num_classes=num_classes, prob_skip=prob_skip)


def ResNet152(prob_skip=0.5, num_classes=10):
    return ResNet(ResidualBlock, [3, 8, 36, 3], num_classes=num_classes, prob_skip=prob_skip)


def train_with_stochastic_depth(model, train_loader, optimizer, criterion, device):
    model.train()
    losses = []
    accuracies = []

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        accuracy = correct / targets.size(0)
        accuracies.append(accuracy)

    return losses, accuracies

import torch

def save_model(model, filepath):
    """
    Save the model's state dictionary to the specified filepath.
    Args:
        model: The PyTorch model to save.
        filepath: The file path where the model will be saved.
    """
    torch.save(model.state_dict(), filepath)
