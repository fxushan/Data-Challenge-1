from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

X_train = np.load('D:/DC1/DataChallenge1/dc1/data/X_train.npy')
Y_train = np.load('D:/DC1/DataChallenge1/dc1/data/Y_train.npy')
X_train = X_train.reshape(X_train.shape[0], 128, 128, 1)

data_gen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.4, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
)

for numb in range(5):
    augmented_images = data_gen.flow(X_train, Y_train, batch_size=1, seed=numb)

    # Initialize arrays to store augmented images and their labels
    X_augmented = np.empty_like(X_train)
    Y_augmented = np.empty_like(Y_train)

    for i in range(X_train.shape[0]):
        augmented_image, label = next(augmented_images)
        X_augmented[i] = augmented_image
        Y_augmented[i] = label

    np.save('D:/DC1/DataChallenge1/dc1/data/X_aug{}.npy'.format(numb), X_augmented)
    np.save('D:/DC1/DataChallenge1/dc1/data/Y_aug{}.npy'.format(numb), Y_augmented)
