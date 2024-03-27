
import torch
import torch.nn as nn


class Net(nn.Module):
    # Leaky ReLU slope_1: float = 0.6807524246386062, slope_2: float = 0.053947204697184746, slope_3: float = 0.5569877536243514
    # PReLU alpha_1=0.7944129993352669, alpha_2=0.340813568624628, alpha_3=0.37811389136107976

    def __init__(self, n_classes: int = 6, alpha_1=1.0887779628058707, alpha_2=1.6631499652552382,
                 alpha_3=1.9301193111245831,
                 p1=0.01, p2=0.21959590088564337, p3=0.26732333432484356) -> None:
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(alpha=alpha_1),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(p=p1),
            nn.Conv2d(64, 32, kernel_size=4, stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(alpha=alpha_2),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout(p=p2),
            nn.Conv2d(32, 16, kernel_size=4, stride=1),
            nn.BatchNorm2d(16),
            nn.ELU(alpha=alpha_3),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=p3),
            # nn.BatchNorm2d(16),
            # nn.ELU(alpha=alpha_4),
            # nn.MaxPool2d(kernel_size=1),
            # nn.Dropout(p=p4),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(144, 256),
            nn.Linear(256, n_classes)
        )


    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x
        elif x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)

        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.reshape(x.size(0), -1)

        x = self.linear_layers(x)
        return x
