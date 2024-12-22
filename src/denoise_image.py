import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_and_preprocess_image(image_path):
    """
    Reads an image from disk, resizes to 32x32, converts to RGB,
    and normalizes to [0,1].
    Returns: (1, 32, 32, 3) float32 numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read the image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def add_gaussian_noise(image_batch, noise_factor=0.5):
    """
    Adds Gaussian noise to images.
    image_batch: shape (N, 32, 32, 3), assumed in [0,1].
    noise_factor: float controlling intensity of the noise.
    """
    noisy = image_batch + noise_factor * np.random.normal(loc=0.0,
                                                          scale=1.0,
                                                          size=image_batch.shape)
    return np.clip(noisy, 0.0, 1.0)


def main(args):
    autoencoder = tf.keras.models.load_model('autoencoder.h5')
    input_img = load_and_preprocess_image(args.image_path)
    if args.add_noise:
        noisy_img = add_gaussian_noise(input_img, noise_factor=args.noise_factor)
    else:
        noisy_img = input_img
    denoised_img = autoencoder.predict(noisy_img)
    original_squeezed = input_img[0]
    noisy_squeezed = noisy_img[0]
    denoised_squeezed = denoised_img[0]

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    axs[0].imshow(original_squeezed)
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(noisy_squeezed)
    axs[1].set_title("Noisy")
    axs[1].axis('off')

    axs[2].imshow(denoised_squeezed)
    axs[2].set_title("Denoised")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Denoise a single image using the trained autoencoder.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to denoise.')
    parser.add_argument('--add_noise', action='store_true', help='Add Gaussian noise before denoising.')
    parser.add_argument('--noise_factor', type=float, default=0.5, help='Intensity of Gaussian noise.')
    args = parser.parse_args()

    main(args)
