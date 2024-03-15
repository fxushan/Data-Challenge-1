
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_classes: int = 6, slope_1: float = 0.6807524246386062,
                 slope_2: float = 0.053947204697184746, slope_3: float = 0.5569877536243514) -> None:
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=slope_1),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 32, kernel_size=4, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=slope_2),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 16, kernel_size=4, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=slope_3),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.125),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(144, 256),
            nn.Linear(256, n_classes)
        )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
