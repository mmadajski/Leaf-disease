import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Loads and resizes an image by given path.
    :param image_path: Path to the image.
    :return: Image in a numpy array with dimensions [256, 256, 3].
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, [256, 256])

    return image


def load_mask(mask_path: str) -> np.ndarray:
    """
    Similar to the "load_image" function, it loads an image mask and resizes it to [256, 256].
    In this dataset, the masks are black and red, so this function works only on the red channel.
    :param mask_path: Path to the mask.
    :return: Mask in a numpy array with dimensions [256, 256].
    """
    mask = cv2.imread(mask_path)[:, :, 2]
    mask = cv2.resize(mask, [256, 256])
    mask_binary = np.where((mask > 64), 255, 0)

    return mask_binary
