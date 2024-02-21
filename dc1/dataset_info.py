import torch

from dc1.image_dataset import ImageDataset
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image
#Images are 1024x1024 px originally
#Scaled down to 128x128 px in our database to reduce size

#Dataset itself is in Numpy arrays, which was originally Pytorch
train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))

print(train_dataset.imgs[0])
print(train_dataset.targets[0])
#It is split between images and targets(labels), they are in cat codes, order is yet to be seen
tensor = torch.from_numpy(train_dataset.imgs[0] / 255).float()
print(tensor.size())

#The images are in color
image = tensor.squeeze(0)
plt.imshow(image, interpolation='nearest')
plt.show()