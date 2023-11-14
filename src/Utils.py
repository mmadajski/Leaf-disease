import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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


def calculate_iou(model: tf.keras.Model, images: np.ndarray, masks: np.ndarray) -> float:
    """
    Calculates and returns the iou metric for a segmentation model.
    :param model: Model to be used in segmentation.
    :param images: Input images.
    :param masks: True masks.
    :return: The value of the iou metric for a given dataset.
    """

    iou_metric = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])
    prediction = model.predict(images)
    prob_to_class = tf.map_fn(fn=lambda x: int(x > 0.5), elems=prediction, dtype="int32")
    iou = iou_metric(prob_to_class, masks)

    return iou


def save_samples(model: tf.keras.Model, images: np.ndarray, masks: np.ndarray, path: str) -> None:
    """
    Saves sample images in the specified path.

    :param model: Model to be used in segmentation.
    :param images: Input images.
    :param masks: True masks.
    """
    predicted_mask = (model.predict(images) * 255).astype("uint8")

    for i, image in enumerate(images):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        image = (images[i] * 255).astype("uint8")
        mask = (masks[i] * 255).astype("uint8")

        ax1.set_title("Sample image")
        ax1.imshow(image)
        ax1.axis("off")

        ax2.set_title("Predicted mask")
        ax2.imshow(predicted_mask[i], cmap="Greys")
        ax2.axis("off")

        ax3.set_title("True mask")
        ax3.imshow(mask, cmap="Greys")
        ax3.axis("off")

        plt.savefig(path + "\\Sample_" + str(i) + ".png")