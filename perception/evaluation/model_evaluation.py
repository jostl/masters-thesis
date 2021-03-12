from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from perception.custom_datasets import ComparisonDataset


def rmse(targets, predictions):
    """
    Depth estimation evaluation method. Average Root Mean Squared Error for each pixel in an image.
    Should work for batches and single images.
    """
    return np.sqrt(np.average((targets-predictions)**2))


def accuracy_within_threshold(targets, predictions, threshold=1.25):
    """
    Depth estimation evaluation method. Calculates a delta value for each pixel in an image, and
    then checks which percentage of pixels are within a certain threshold.
    Should work for batches and single images.
    """

    targets_over_preds = targets / predictions
    preds_over_targets = predictions / targets

    deltas = np.maximum(targets_over_preds, preds_over_targets)

    within_threshold_matrix = deltas < threshold

    _, counts = np.unique(within_threshold_matrix, return_counts=True)

    accuracy = counts[1] / (counts[0] + counts[1])  # this will work as long as there are both True and False values

    return accuracy


def compare_models(data_folder, segmentation_models, depth_models, batch_size=10):
    targets = ComparisonDataset(data_folder, segmentation_models, depth_models, max_n_instances=15)

    dataloader = DataLoader(targets, batch_size=batch_size, shuffle=False, num_workers=0,
                            pin_memory=True)

    # semantic segmentation metrics
    # TODO add semantic segmentation metrics here

    # depth estimation metrics
    accuracy_with_threshold_accumulated = defaultdict(int)
    rmse_accumulated = defaultdict(int)

    count_batches = 0
    for rgb_targets, segmentation_targets, depth_targets, segmentation_preds, depth_preds in dataloader:
        count_batches += 1
        for model in segmentation_preds:
            pass  # TODO

        for model in depth_preds:
            accuracy_with_threshold_accumulated[model] += accuracy_within_threshold(depth_targets, depth_preds[model])
            rmse_accumulated[model] += rmse(depth_targets, depth_preds[model])

    n_batches = np.ceil(len(targets) / batch_size)
    assert n_batches == count_batches  # TODO fjern count_batches

    accuracy_with_threshold_avg = {}
    rmse_accumulated_avg = {}
    for model in depth_models:
        model_name = model[0]
        accuracy_with_threshold_avg[model_name] = accuracy_with_threshold_accumulated[model_name] / n_batches
        rmse_accumulated_avg[model_name] = rmse_accumulated[model_name] / n_batches

    print("finished comparison, this print is here for debugging reasons")
    # TODO speed measurement


if __name__ == "__main__":

    # location of where to find training, test1, test2
    data_folder = Path("data/perception/train_10k/train")
    predictions_folder = Path("data/perception/predictions")

    # lagres på formatet (Navn, lokasjon)
    segmentation_models = [("nvidia-test", predictions_folder / "nvidia_test")]
    depth_models = [("MiDaS-test", predictions_folder / "midas_test")]

    compare_models(data_folder, segmentation_models, depth_models)

