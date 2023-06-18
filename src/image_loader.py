import os
from typing import Optional

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tqdm import tqdm


def get_image_paths(folder: str, start: int, end: int):
    """
    Get slice of image paths in a folder

    :param folder: Folder containing images
    :param start: Start index
    :param end: End index
    """
    return [os.path.join(folder, file_name) for file_name in os.listdir(folder)[start:end]]

def read_images(file_paths: list, *, resize_dimensions: Optional[tuple] = (400,400), show_progress: bool = True):
    """
    Read images from file paths

    :param file_paths: List of file paths
    :param resize_dimensions: Dimensions to resize images to
    :param show_progress: Show progress bar
    """

    images = []
    folder_contents = file_paths
    if show_progress:
        folder_contents = tqdm(file_paths)

    for filename in folder_contents:
        img = Image.open(filename)
        if resize_dimensions is not None:
            img = ImageOps.fit(img, resize_dimensions)
        img = np.asarray(img) / 255.0
        img = img.reshape((*img.shape, 1))

        images.append(tf.constant(img))
    
    return np.array(images)
