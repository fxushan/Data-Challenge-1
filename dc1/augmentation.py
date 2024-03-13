from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tqdm import tqdm

X_train = np.load('../dc1/data/X_train.npy')
Y_train = np.load('../dc1/data/Y_train.npy')
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

augmented_images = data_gen.flow(X_train, Y_train, batch_size=1)

num_augmented_per_original = 5  # Number of augmented images to generate per original image
X_augmented = []  # To store augmented images
Y_augmented = []  # To store corresponding labels

print(f'Amount of elements to loop over: {len(list(zip(X_train, Y_train)))}')
for x, y in tqdm(zip(X_train, Y_train)):
    for _ in range(num_augmented_per_original):
        img = next(augmented_images)[0]
        img_squeezed = np.squeeze(img, axis=0)  # Remove the batch dimension
        X_augmented.append(img_squeezed)  # Append the image without the batch dimension
        Y_augmented.append(y)  # Same label as the original image

print('Loop done\n')
print('Converting lists to numpy arrays')
# Convert lists to numpy arrays
X_augmented = np.array(X_augmented)
Y_augmented = np.array(Y_augmented)

print('Convert done\n')
print('Appending augmented data to original data')
# Append the augmented data to the original data
X_train_augmented = np.concatenate((X_train, X_augmented), axis=0)
Y_train_augmented = np.concatenate((Y_train, Y_augmented), axis=0)

print('Append done\n')
print('Saving files')
np.save('../dc1/data/X_train_augmented.npy', X_train_augmented)
np.save('../dc1/data/Y_train_augmented.npy', Y_train_augmented)
