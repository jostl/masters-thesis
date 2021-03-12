import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from perception.custom_datasets import ComparisonDataset


def accuracy_with_threshold(targets, predictions, threshold=1.25):
    """
    Depth estimation evaluation method. Calculates a delta value for each pixel in an image, and
    then checks which percentage of pixels are within a certain threshold.
    Should work for batches and single images
    """

    targets_over_preds = targets / predictions
    preds_over_targets = predictions / targets

    deltas = np.maximum(targets_over_preds, preds_over_targets)

    within_threshold_matrix = deltas < threshold

    _, counts = np.unique(within_threshold_matrix, return_counts=True)

    accuracy = counts[1] / (counts[0] + counts[1])  # this will work as long as there are both True and False values

    return accuracy


def compute_depth_loss(targets, predictions):
    pass


def compare_models(data_folder, segmentation_models, depth_models):
    targets = ComparisonDataset(data_folder, segmentation_models, depth_models, n_semantic_classes=6,
                                max_n_instances=15)

    dataloader = DataLoader(targets, batch_size=10, shuffle=False, num_workers=0,
                            pin_memory=True)

    accuracy_with_threshold_accumulated = defaultdict(int)
    for rgb_targets, segmentation_targets, depth_targets, segmentation_preds, depth_preds in dataloader:
        # TODO plot for sammenligning
        for model in segmentation_preds:
            pass

        for model in depth_preds:
             accuracy_with_threshold_accumulated[model] += accuracy_with_threshold(depth_targets, depth_preds[model])




if __name__ == "__main__":

    # location of where to find training, test1, test2
    data_folder = Path("data/perception/train_10k/train")
    predictions_folder = Path("data/perception/predictions")

    # lagres på formatet (Navn, lokasjon)
    segmentation_models = [("nvidia-test", predictions_folder / "nvidia_test")]
    depth_models = [("MiDaS-test", predictions_folder / "midas_test")]

    compare_models(data_folder, segmentation_models, depth_models)

