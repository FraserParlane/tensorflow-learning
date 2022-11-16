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


def plot_test_data(
        model: tf.keras.Model,
        test_x: np.ndarray,
        test_y: np.ndarray,
        n: int = 0,
) -> None:
    """
    Plot the probabilities for a given test value
    :param model: The model to predict with
    :param test_x: The test values
    :param test_y: The train values
    :param n: The index to visualize
    :return: None
    """

    # Predict the value
    prop = model.predict(
        test_x[n:n+1],
        verbose=0,
    )[0]

    # Create the figure
    figure: plt.Figure = plt.figure(
        dpi=300,
        figsize=(4, 8)
    )
    im_ax: plt.Axes = figure.add_subplot(2, 1, 1)
    prob_ax = plt.Axes = figure.add_subplot(4, 1, 3)
    log_prob_ax = plt.Axes = figure.add_subplot(4, 1, 4)

    # Plot image
    im_ax.imshow(
        test_x[n],
        cmap='Greys',
    )

    # Plot prob, log_prob
    x = np.linspace(0, 9, 10)
    bar_format = dict(
        color='black',
    )
    prob_ax.bar(
        x,
        prop,
        **bar_format,
    )

    log_prob_ax.bar(
        x,
        prop,
        **bar_format,
    )

    # Note the label
    y_pos = 1.1
    prob_ax.text(
        0,
        1.1,
        'True label:',
        clip_on=False,
        transform=prob_ax.transAxes,
        horizontalalignment='right',
    )

    # Add circle for label
    prob_ax.scatter(
        test_y[n],
        y_pos + 0.05,
        clip_on=False,
        color='black',
    )

    # Format image
    im_ax.set_xticks([])
    im_ax.set_yticks([])

    # Format bars
    for ax in [prob_ax, log_prob_ax]:
        ax.set_xticks(x)
        for pos in ['top', 'right']:
            ax.spines[pos].set_visible(False)
    y_range = [0, 1]
    prob_ax.set_yticks(y_range)
    prob_ax.set_ylim(y_range)
    log_prob_ax.set_yscale('log')
    prob_ax.set_ylabel('predicted\nprobability')
    log_prob_ax.set_ylabel('log predicted\nprobability')

    # Format figure
    figure.subplots_adjust(
        hspace=0.3,
        left=0.25,
        right=0.85,
        top=0.98,
        bottom=0.1,

    )

    # Save
    folder = os.path.join(os.getcwd(), 'plots')
    if not os.path.exists(folder):
        os.mkdir(folder)
    figure.savefig(os.path.join(folder, f'{str(n).zfill(5)}.png'))


def plot_bad(
        n: int = 10,
):
    """
    Make a plot of all the mislabelled images
    :param n: the number of bad plots to make.
    :return: None
    """

    # Get the data
    train_x, train_y, test_x, test_y = get_data(use_cache=True)

    # Get the model
    model = get_trained_model(
        use_model_cache=True,
    )

    # For each test value
    for i, (i_test_x, i_test_y) in tqdm(
        enumerate(zip(test_x, test_y)),
        total=len(test_x),
    ):

        # Predict
        preds = model.predict(i_test_x[None, :], verbose=0)[0]
        pred = np.argmax(preds)

        # If not right, plot
        if pred != i_test_y:

            # Plot example
            plot_test_data(
                model=model,
                test_x=test_x,
                test_y=test_y,
                n=i,
            )

            # Only perform for n images.
            n -= 1
            if n == 0:
                break


def plot_n(
        n: int = 0,
) -> None:
    """
    Plot
    :param n: Plot the first n pred.
    :return: None
    """

    # Get the data
    train_x, train_y, test_x, test_y = get_data(use_cache=True)

    # Get the model
    model = get_trained_model(
        use_model_cache=True,
    )

    # For the first n images
    for i in tqdm(range(n)):

        # Plot example
        plot_test_data(
            model=model,
            test_x=test_x,
            test_y=test_y,
            n=i,
        )


def run():
    """
    Train the model and plot some examples.
    :return: None
    """
    logging.getLogger().setLevel(logging.INFO)
    model = get_trained_model(
        use_data_cache=False,
        use_model_cache=False,
    )
    plot_n(n=10)
    plot_bad(n=10)


if __name__ == '__main__':
    run()
