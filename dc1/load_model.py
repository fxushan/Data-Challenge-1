from pathlib import Path

from matplotlib import pyplot as plt
from pytorch_grad_cam import HiResCAM

import torch
from dc1.image_dataset import ImageDataset
from dc1.net import Net

train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
model = Net(n_classes=6)
model.load_state_dict(torch.load(Path("../dc1/model_weights/model_03_20_14_11.pth")))
model.eval()
target_layers = [model.cnn_layers[-1]]
cam = HiResCAM(model, target_layers)
input_tensor = torch.from_numpy(train_dataset.imgs[0] / 255).float()
input_tensor = input_tensor.unsqueeze(0)

grayscale_cam = cam(input_tensor=input_tensor)

image = input_tensor.squeeze(0).squeeze(0)

# Plot the original image
plt.imshow(image.numpy(), cmap='gray')

torch_2 = torch.from_numpy(grayscale_cam / 255).float()
image_2 = torch_2.squeeze(0)
# Overlay the heatmap gradient
plt.imshow(image_2, cmap='hot', alpha=0.5)  # Adjust alpha to control transparency

# Add color bar for the heatmap gradient
plt.colorbar()

# Show the plot
plt.show()
