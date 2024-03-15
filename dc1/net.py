import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(p=0.5, inplace=True),
            nn.Conv2d(64, 32, kernel_size=4, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout(p=0.25, inplace=True),
            nn.Conv2d(32, 16, kernel_size=4, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.125, inplace=True),
        )

        self.flatten_size = 144 # This needs to be the same as before, determined by the output size of your last conv layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.flatten_size, self.flatten_size),
            nn.Tanh(),
            nn.Linear(self.flatten_size, 1),
            nn.Softmax(dim=1)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(144, 256),
            nn.Linear(256, n_classes)
        )

        self.last_attention_weights = None  # Store the last attention weights here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1) # Flatten the output for the attention mechanism

        # Apply attention
        attention_weights = self.attention_layer(x)
        self.last_attention_weights = attention_weights # Store the attention weights
        x = x * attention_weights.expand_as(x)

        x = self.linear_layers(x)
        return x
