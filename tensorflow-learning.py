import matplotlib.pyplot as plt
from typing import Tuple
import tensorflow as tf
from tqdm import tqdm
import urllib.request
import pandas as pd
import numpy as np
import logging
import shutil
import dill
import os


def get_data(
        use_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the training and testing data
    :param use_cache: Use local cache for speed
    :return: train_x, train_y, test_x, test_y
    """

    # Where the cached data will be stored
    data_dir = os.path.join(os.getcwd(), 'data.dill')

    # If using the cache
    logging.info('Loading data')
    if use_cache and os.path.exists(data_dir):

        # Load data
        logging.info('Using cache')
        with open(data_dir, 'rb') as f:
            train_x, train_y, test_x, test_y = dill.load(f)

    # If not using the cache
    else:

        # Get the training data from Keras
        logging.info('Not using cache')
        from keras.datasets import mnist
        (train_x, train_y), (test_x, test_y) = mnist.load_data()

        # Delete old data if present
        if os.path.exists(data_dir):
            os.remove(data_dir)

        # Save to disk
        with open(data_dir, 'wb') as f:
            dill.dump((train_x, train_y, test_x, test_y), f)

    # Return
    return train_x, train_y, test_x, test_y


def make_model() -> tf.keras.models.Sequential:
    """
    Make the CNN model for training
    :return: The model.
    """

    # Make the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='he_uniform',
        input_shape=(28, 28, 1),
    ))
    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
    ))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=100,
        activation='relu',
        kernel_initializer='he_uniform',
    ))
    model.add(tf.keras.layers.Dense(
        units=10,
        activation='softmax',
    ))

    # Return
    return model


def train_cnn(
) -> None:
    """
    Create the CNN and train it on the data
    :return:
    """

    # Get the training data from Keras. Cached locally after first load.
    # 60,000 in train, 10,000 in test. images are 28 x 28.
    train_x, train_y, test_x, test_y = get_data(use_cache=True)

    # Normalize the x data to be between 0 and 1.
    train_x = train_x / 255
    test_x = test_x / 255

    # Make the CNN model
    model = make_model()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    train_cnn()
