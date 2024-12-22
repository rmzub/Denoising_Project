import argparse
import tensorflow as tf
from model import build_cifar10_autoencoder

def load_cifar10():
    """
    Loads CIFAR-10 data, normalizes it to [0,1].
    Returns:
        (x_train, x_test): both in float32, shape (N, 32, 32, 3)
    """
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, x_test


def main(args):
    x_train, x_test = load_cifar10()
    autoencoder = build_cifar10_autoencoder()
    autoencoder.summary()
    autoencoder.fit(
        x_train, x_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        validation_data=(x_test, x_test)
    )

    autoencoder.save('autoencoder.h5')
    print("Model saved to autoencoder.h5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the autoencoder on CIFAR-10.")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20).')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128).')
    args = parser.parse_args()

    main(args)
