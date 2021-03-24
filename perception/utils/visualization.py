import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from perception.utils.segmentation_labels import CARLA_CLASSES, DEFAULT_CLASSES


def get_segmentation_colors(n_classes, only_random=False, class_indxs=None, color_seed=73):
    assert only_random or class_indxs
    random.seed(color_seed)
    class_colors = []
    if only_random:
        for _ in range(n_classes):
            class_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        return class_colors
    elif class_indxs:
        for c in class_indxs:
            class_colors.append(CARLA_CLASSES[c][1])
        class_colors.append((0, 0, 0))
    return class_colors


def get_rgb_segmentation(semantic_image: np.ndarray, class_colors):
    """
    Creates a RGB image from a semantic image. Semantic image must have shape: (Height, Width, #Semantic Classes)
    """
    height, width, n_classes = semantic_image.shape
    semantic_image_rgb = np.zeros((height, width, 3))
    semantic_pred_argmax = semantic_image.argmax(axis=2)
    for c in range(n_classes):
        semantic_image_rgb[:, :, 0] += ((semantic_pred_argmax[:, :] == c) * (class_colors[c][0])).astype('uint8')
        semantic_image_rgb[:, :, 1] += ((semantic_pred_argmax[:, :] == c) * (class_colors[c][1])).astype('uint8')
        semantic_image_rgb[:, :, 2] += ((semantic_pred_argmax[:, :] == c) * (class_colors[c][2])).astype('uint8')
    return semantic_image_rgb


def display_images_horizontally(images, fig_width, fig_height):
    # Inspired from Hands-On Machine Learning with SciKit-learn, Keras and TensorFlow, page 574
    # Displays the list of images horizontally.

    def plot_image(image, cmap="binary"):
        # todo: https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa
        plt.imshow(image, cmap=cmap)
        plt.axis("off")
        # plt.show()

    n_images = len(images)
    if n_images > 0:
        fig = plt.figure(figsize=(fig_width, fig_height))
        for image_index in range(n_images):
            image = images[image_index]
            plt.subplot(1, n_images, 1 + image_index)
            cmap = "binary" if len(images[image_index].shape) == 3 else "gray"
            plot_image(image, cmap=cmap)
        fig.show()


def display_originals_with_decoded(original_images, decoded_images, title=""):
    # Inspired by Hands-On Machine Learning with SciKit-learn, Keras and TensorFlow, page 574.
    # Meant to be used for visualization of target images and predicted images in multi-task learning.
    # Target images displayed in top row, predicted images in row below.

    def plot_image(image, cmap="binary"):
        # todo: https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa
        plt.imshow(image, cmap=cmap)
        plt.axis("off")
        # plt.show()

    n_images = len(original_images)
    if n_images > 0:
        fig = plt.figure(figsize=(n_images * 1.2, 3))
        fig.suptitle(title, fontsize=10)
        for image_index in range(n_images):
            cmap = "binary" if original_images[image_index].shape[-1] == 3 else "gray"
            plt.subplot(2, n_images, 1 + image_index)
            plot_image(original_images[image_index], cmap=cmap)
            plt.subplot(2, n_images, 1 + n_images + image_index)
            plot_image(decoded_images[image_index], cmap=cmap)
        fig.show()


def show_predictions(model, inputs, device, semantic_classes, n_displays=1, title=""):
    # input_image has size (Height, Width, N-Channels).
    # Have to add batch dimension, and transpose it to able to make predictions
    rgb_inputs, rgb_targets, semantic_targets, depth_targets = inputs[0].to(device), inputs[1].to(device), inputs[2].to(
        device), inputs[3].to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(rgb_inputs)
    # Send all predictions and target tensors to cpu
    n_displays = min(n_displays, len(rgb_inputs))

    rgb_preds, semantic_preds, depth_preds = [pred.cpu().numpy().transpose(0, 2, 3, 1)[:n_displays] for pred in
                                              predictions]
    rgb_targets = rgb_targets.cpu().numpy().transpose(0, 2, 3, 1)[:n_displays]
    depth_targets = depth_targets.cpu().numpy().transpose(0, 2, 3, 1)[:n_displays]
    semantic_targets = semantic_targets.cpu().numpy().transpose(0, 2, 3, 1)[:n_displays]
    for i in range(n_displays):
        rgb_pred = rgb_preds[i]
        semantic_pred = semantic_preds[i]
        depth_pred = depth_preds[i]

        rgb_target = rgb_targets[i]
        semantic_target = semantic_targets[i]
        depth_target = depth_targets[i]

        class_colors = get_segmentation_colors(semantic_classes, class_indxs=semantic_classes)

        semantic_pred_rgb = get_rgb_segmentation(semantic_image=semantic_pred, class_colors=class_colors)
        semantic_target_rgb = get_rgb_segmentation(semantic_image=semantic_target, class_colors=class_colors)

        semantic_pred_rgb = semantic_pred_rgb / 255
        semantic_target_rgb = semantic_target_rgb / 255

        # Setup original images for display
        original_images = [rgb_target, semantic_target_rgb, depth_target]
        # Setup decoded images for display
        decoded_images = [rgb_pred, semantic_pred_rgb, depth_pred]

        # Show rgb, semantic segmentation and depth images with corresponding predictions
        display_originals_with_decoded(original_images=original_images, decoded_images=decoded_images, title=title)


def plot_image(image, title="", cmap="binary"):
    # todo: https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.show()


def plot_segmentation(image: np.ndarray):
    _, _, n_classes = image.shape
    class_colors = get_segmentation_colors(n_classes=n_classes, class_indxs=DEFAULT_CLASSES)
    semantic_image_rgb = get_rgb_segmentation(image, class_colors=class_colors) / 255
    plot_image(semantic_image_rgb)
    plt.show()
