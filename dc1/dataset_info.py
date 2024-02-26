import torch

from dc1.image_dataset import ImageDataset
from pathlib import Path
from matplotlib import pyplot as plt
#Images are 1024x1024 px originally
#Scaled down to 128x128 px in our database to reduce size

#Dataset itself is in Numpy arrays, which was originally Pytorch
train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))

print(train_dataset.imgs[1])
print(train_dataset.targets[0])

'''
It is split between images and targets(labels), they are in cat codes, order is:
1. Pneumothorax
2. Nodule
3. No Finding
4. Infiltration
5. Effusion
6. Atelectasis
'''

torch = torch.from_numpy(train_dataset.imgs[1] / 255).float()
print(torch)

#The images are gray, the dimension 1,128,128 means that it is gray.
image = torch.squeeze(0)
plt.imshow(image.numpy(), cmap='gray', interpolation='nearest')
plt.show()