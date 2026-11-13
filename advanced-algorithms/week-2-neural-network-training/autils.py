from typing import Union as _Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.figure import Figure as _Figure
from numpy.typing import NDArray as _NDArray


def load_data(file_path: str) -> tuple[_NDArray[np.float64], _NDArray[np.uint8]]:
    """Load feature and label arrays from matching ``X.npy`` and ``y.npy`` files.

    The label file path is derived by replacing ``X`` with ``y`` in
    ``file_path``. This dataset stores features as ``float64`` and labels as
    ``uint8``; all rows in both files are returned.

    Returns:
        The feature matrix and corresponding label matrix.
    """
    X = np.load(file_path)
    y = np.load(file_path.replace('X', 'y'))
    return X, y


def sigmoid(
    z: _Union[_NDArray[np.float64], int, float],
) -> _Union[_NDArray[np.float64], np.float64]:
    """Compute the logistic sigmoid for a NumPy array or numeric scalar."""
    return 1 / (1 + np.exp(-z))


def widgvis(fig: _Figure) -> None:
    """Hide the toolbar, header, and footer on a widget-backed figure."""
    canvas = fig.canvas
    for attr in ('toolbar_visible', 'header_visible', 'footer_visible'):
        if hasattr(canvas, attr):
            setattr(canvas, attr, False)


def plot_loss_tf(history: tf.keras.callbacks.History) -> None:
    """Plot the per-epoch training loss recorded by ``model.fit``."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    widgvis(fig)
    ax.plot(history.history['loss'], label='loss')
    ax.set_ylim((0, 2))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss (cost)')
    ax.legend()
    ax.grid(True)
    plt.show()


def display_digit(X: _NDArray[np.float64]) -> None:
    """Display one digit image stored as a flattened 400-pixel array."""
    fig, ax = plt.subplots(1, 1, figsize=(0.5, 0.5))
    widgvis(fig)
    image = X.reshape((20, 20)).T
    ax.imshow(image, cmap='gray')
    plt.show()


def display_errors(
    model: tf.keras.Model, X: _NDArray[np.float64], y: _NDArray[np.uint8]
) -> int:
    """Count misclassified images and plot up to eight of the mistakes.

    ``X`` contains flattened 400-pixel images, and ``y`` contains their
    integer labels in its first column.

    Returns:
        The number of misclassified images.
    """
    logits = model.predict(X)
    predicted_labels = np.argmax(logits, axis=1)
    error_indices = np.where(predicted_labels != y[:, 0])[0]
    if len(error_indices) == 0:
        print("no errors found")
    else:
        display_count = min(8, len(error_indices))
        fig, axes = plt.subplots(
            1, display_count, figsize=(5, 1.2), squeeze=False
        )
        fig.tight_layout(pad=0.13, rect=(0, 0.03, 1, 0.80))
        widgvis(fig)

        for i in range(display_count):
            image_index = error_indices[i]
            image = X[image_index].reshape((20, 20)).T
            axes[0, i].imshow(image, cmap='gray')
            axes[0, i].set_title(
                f"{y[image_index, 0]},{predicted_labels[image_index]}",
                fontsize=10,
            )
            axes[0, i].set_axis_off()
            fig.suptitle("Label, yhat", fontsize=12)
    return len(error_indices)