import tensorflow as tf
from tensorflow.keras import layers, models

def build_cifar10_autoencoder():
    """
    Builds a convolutional autoencoder for CIFAR-10 (32x32 RGB images).
    Returns:
        A compiled Keras model ready for training.
    """
    inputs = tf.keras.Input(shape=(32, 32, 3))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Output: 16x16
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Output: 8x8

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)  # Still 8x8

    x = layers.UpSampling2D((2, 2))(x)  # Back to 16x16
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # Back to 32x32
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder
