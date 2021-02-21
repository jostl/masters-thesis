import glob

import cv2
import numpy as np

import os
def read_images(folder_name: str, normalize: bool = False, file_format="png",
                max_n_instances=None) -> np.ndarray:
    """
    Reads RGB images and outputs numpy array in BGR format.

    :param folder_name: Path to folder containing images
    :param normalize: Set to 'True' if images should be normalized. 'False' is default.
    :return: numpy array containing images
    """
    array = get_numpy_image_array(folder_name, file_format=file_format, max_n_instances=max_n_instances)
    array = array / 255 if normalize else array
    return array


def get_numpy_image_array(folder_name, file_format="png", max_n_instances=None):
    files = glob.glob(folder_name + "/*." + file_format)
    end_range = min((len(files), max_n_instances)) if max_n_instances is not None else len(files)
    return np.array([cv2.imread(files[i]) for i in range(end_range)], dtype="float32")


def read_segmentation_images(folder_name: str, n_classes, file_format="png", max_n_instances=None) -> np.ndarray:
    array = get_numpy_image_array(folder_name=folder_name, file_format=file_format, max_n_instances=max_n_instances)
    n_instances, height, width, _ = array.shape
    segmentation_labels = np.zeros((n_instances, height, width, n_classes))
    # TODO: Bruk heller metoden i helpers.py?
    for i in range(len(array)):
        image = array[i]
        image = image[:, :, 0]
        for c in range(n_classes):
            segmentation_labels[i, :, :, c] = (image == c).real
    return segmentation_labels


def read_depth_images(folder_name: str, max_n_instances=None, file_format="jpeg") -> np.ndarray:
    # TODO: undersøk dette kanskje?
    array = get_numpy_image_array(folder_name=folder_name, max_n_instances=max_n_instances, file_format=file_format)
    n_instances, height, widht, _ = array.shape
    depth_array = np.zeros((n_instances, height, widht, 1))
    for i in range(len(array)):
        depth_array[i] = get_depth_array(array[i], width=widht, height=height)
    return depth_array


def get_depth_array(image_input, width, height):
    """ Load depth array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    else:
        if not os.path.isfile(image_input):
            raise Exception("get_segmentation_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    img = img[:, :, 0]
    img = np.reshape(img, (width, height, 1))

    return (img / 255) * 2 - 1
