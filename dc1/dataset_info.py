import torch

from dc1.image_dataset import ImageDataset
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
#Images are 1024x1024 px originally
#Scaled down to 128x128 px in our database to reduce size

#Dataset itself is in Numpy arrays, which was originally Pytorch
train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))

print(train_dataset.imgs[1])
print(train_dataset.targets[0])

'''
It is split between images and targets(labels), they are in cat codes, order is:
0. Pneumothorax
1. Nodule
2. No Finding
3. Infiltration
4. Effusion
5. Atelectasis
'''

torch = torch.from_numpy(train_dataset.imgs[1] / 255).float()
print(torch)

#The images are gray, the dimension 1,128,128 means that it is gray.
image = torch.squeeze(0)
plt.imshow(image.numpy(), cmap='gray', interpolation='nearest')
plt.show()

print(np.unique(train_dataset.targets))
