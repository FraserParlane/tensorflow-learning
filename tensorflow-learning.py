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
        logging.info('Using data cache')
        with open(data_dir, 'rb') as f:
            train_x, train_y, test_x, test_y = dill.load(f)

    # If not using the cache
    else:

        # Get the training data from Keras
        logging.info('Not using data cache')
        from keras.datasets import mnist
        (train_x, train_y), (test_x, test_y) = mnist.load_data()

        # Normalize
        train_x = train_x / 255
        test_x = test_x / 255

        # Delete old data if present
        if os.path.exists(data_dir):
            os.remove(data_dir)

        # Save to disk
        with open(data_dir, 'wb') as f:
            dill.dump((train_x, train_y, test_x, test_y), f)

    # Return
    return train_x, train_y, test_x, test_y


def make_model() -> tf.keras.Model:
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


def train_model(
        model: tf.keras.Model,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
) -> tf.keras.Model:
    """
    Create the CNN and train it on the data
    :return: the model
    """

    # Compile the model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )

    # Train the model
    logging.info('Training model')
    model.fit(
        x=train_x,
        y=train_y,
        epochs=4,
        validation_data=(test_x, test_y),
    )

    # Save the model
    model_path = os.path.join(os.getcwd(), 'model')
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    model.save(model_path)

    # Return the trained model
    return model


def get_trained_model(
        use_data_cache: bool = True,
        use_model_cache: bool = True,
) -> tf.keras.Model:

    # If using cache, return
    model_path = os.path.join(os.getcwd(), 'model')
    if use_model_cache and os.path.exists(model_path):
        logging.info('Loading cached model')
        model = tf.keras.models.load_model(model_path)
        return model

    # Don't use cache
    else:

        # Get data
        train_x, train_y, test_x, test_y = get_data(
            use_cache=use_data_cache
        )

        # Get model
        model = make_model()

        # Train model
        model = train_model(
            model=model,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
        )

        # Return trained model
        return model


def run():

    # Get the data
    train_x, train_y, test_x, test_y = get_data(use_cache=True)

    # Get the model
    model = get_trained_model(
        use_model_cache=True,
    )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
