import tensorflow as tf
import numpy as np

X_train = np.load('../dc1/data/X_train.npy')
Y_train = np.load('../dc1/data/Y_train.npy')

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])


def augment_data(images, labels, augmentation_pipeline, num_augmented_per_original=5):
    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):
        for _ in range(num_augmented_per_original):
            image_expanded = np.expand_dims(image, axis=0)
            augmented_image = augmentation_pipeline(image_expanded)
            augmented_image = np.squeeze(augmented_image.numpy(), axis=0)
            augmented_images.append(augmented_image)
            augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)


X_augmented, Y_augmented = augment_data(X_train, Y_train, data_augmentation)

print('Concatenating')
X_combined = np.concatenate((X_train, X_augmented), axis=0)
Y_combined = np.concatenate((Y_train, Y_augmented), axis=0)

print('Saving files')
np.save('../dc1/data/X_tf.npy', X_combined)
np.save('../dc1/data/Y_tf.npy', Y_combined)
