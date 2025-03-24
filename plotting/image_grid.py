import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def plot_image_grid(images: list[np.ndarray | Image.Image], ncols: int, width: float = 8., title: str = None, labels: list[str] = None) -> None:
    """Plots a collection of images as a rectangular grid with a given number of columns.

    Args:
        images (list[np.ndarray | Image.Image]): Collection of images.
        ncols (int): Number of columns.
        width (float): Width of the figure in inches.
    """
    assert labels is None or len(labels) == len(images)

    # Calculate number of rows and figure height required
    nrows = math.ceil(len(images) / ncols)
    height = width * nrows / ncols

    plt.figure(figsize=(width, height))

    if title:
        plt.suptitle(title)

    for i, image in enumerate(images):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(image)
        plt.axis('off')

        if labels:
            plt.title(label=labels[i])

    plt.show()
    


if __name__ == '__main__':
    plot_image_grid(
        images=np.random.random((3, 256, 256)), 
        ncols=2, 
        title="This is a title", 
        labels=['test test', 'okay', 'bye']
        )