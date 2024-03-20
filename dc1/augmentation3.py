import imgaug.augmenters as iaa
import numpy as np

X_train = np.load('../dc1/data/X_train.npy')
Y_train = np.load('../dc1/data/Y_train.npy')

seq = iaa.Sequential([
    iaa.Fliplr(0.1),  # horizontally flip 10% of the images
    iaa.Affine(
        rotate=(-10, 10),  # rotate degrees
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}  # scale images independently
    ),
    iaa.Multiply((0.9, 1.1)),  # change brightness
    iaa.LinearContrast((0.9, 1.1)),  # change contrast
    iaa.GaussianBlur(sigma=(0, 1.0)),  # Gaussian blur with sigma of 0 to 1.0
    iaa.AdditiveGaussianNoise(scale=0.02*255)  # add Gaussian noise
], random_order=True)  # apply augmenters in random order

X_augmented = seq(images=X_train)

print('Concatenating')
X_combined = np.concatenate((X_train, X_augmented), axis=0)
Y_combined = np.concatenate((Y_train, Y_train), axis=0)

print('Saving files')
np.save('../dc1/data/X_ia.npy', X_combined)
np.save('../dc1/data/Y_ia.npy', Y_combined)
