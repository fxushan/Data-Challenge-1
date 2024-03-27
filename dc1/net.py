import torch
import torch.nn as nn

import random

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.5):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training:
            return x

        if random.random() < self.drop_prob:
            return x

        return torch.zeros_like(x)


class Net(nn.Module):
    # Leaky ReLU slope_1: float = 0.6807524246386062, slope_2: float = 0.053947204697184746, slope_3: float = 0.5569877536243514
    # PReLU alpha_1=0.7944129993352669, alpha_2=0.340813568624628, alpha_3=0.37811389136107976

    def __init__(self, n_classes: int = 6, alpha_1=1.0887779628058707, alpha_2=1.6631499652552382,
                 alpha_3=1.9301193111245831,
                 p1=0.01, p2=0.21959590088564337, p3=0.26732333432484356) -> None:
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ELU(alpha=alpha_1),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(p=0.5),

            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ELU(alpha=alpha_2),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ELU(alpha=alpha_3),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.125),
        )

        # Calculate the size of the output from convolutional layers
        self.flatten_size = self._get_flatten_size()

        self.linear_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 512),  # Adjusted input size
            nn.ReLU(inplace=True),
            nn.Linear(512, n_classes)
        )

    def _get_flatten_size(self):
        # Define a dummy input tensor to calculate the output size of the convolutional layers
        input_tensor = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            features = self.cnn_layers(input_tensor)
        # Calculate the total number of features after flattening
        return features.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x
        elif x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)

        x = self.cnn_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_layers(x)
        return x
