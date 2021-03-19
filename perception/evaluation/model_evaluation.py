from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from perception.custom_datasets import ComparisonDataset
from perception.utils.visualization import plot_segmentation, plot_image


def jaccard_index(target, predictions):
    """
    Semantic segmentation metric. Calculates mean intersection over union over all classes.
    Works for batches of data.
    """

    intersection = torch.logical_and(target, predictions)
    union = torch.logical_or(target, predictions)

    intersection_sums = torch.sum(intersection, dim=(-2, -1))
    union_sums = torch.sum(union, dim=(-2,-1))

    class_exists_mask = union_sums != 0

    # union_sums will contain 0's if a class is not present in an image, which will give division by zero
    iou_scores_classwise = intersection_sums / (union_sums + 0.00000000001)

    iou_scores_imagewise_sum = iou_scores_classwise.sum(dim=1)
    class_exists_mask_sum = class_exists_mask.sum(dim=1)
    iou_scores_imagewise_mean = iou_scores_imagewise_sum / class_exists_mask_sum

    iou_score_batch_mean = torch.mean(iou_scores_imagewise_mean)

    return iou_score_batch_mean.numpy()


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

    uniques, counts = np.unique(within_threshold_matrix, return_counts=True)

    if len(counts) > 1:
        accuracy = counts[1] / (counts[0] + counts[1])  # this will work as long as there are both True and False values
    else:
        print("Accuracy within threshold warning: Accuracy is ", end="")
        if True in uniques:
            accuracy = 1.
            print("1. uniques:", uniques)
        else:
            accuracy = 0.
            print("0. uniques:", uniques)
    return accuracy


def compare_models(data_folder, segmentation_models, depth_models, batch_size=10):
    targets = ComparisonDataset(data_folder, segmentation_models, depth_models, max_n_instances=150)  # TODO sett opp max_n

    dataloader = DataLoader(targets, batch_size=batch_size, shuffle=False, num_workers=0,
                            pin_memory=True)

    # semantic segmentation metrics
    mean_intersection_over_union_accumulated = defaultdict(int)

    # depth estimation metrics
    accuracy_with_threshold_accumulated = defaultdict(int)
    accuracy_with_threshold2_accumulated = defaultdict(int)
    accuracy_with_threshold3_accumulated = defaultdict(int)
    rmse_accumulated = defaultdict(int)

    for rgb_targets, segmentation_targets, depth_targets, segmentation_preds, depth_preds in tqdm(dataloader):

        for model in segmentation_preds:
            mean_intersection_over_union_accumulated[model] \
                += jaccard_index(segmentation_targets, segmentation_preds[model])

            #img = segmentation_preds[model].numpy()[0].transpose(1, 2, 0)
            #plot_segmentation(img)

        for model in depth_preds:
            accuracy_with_threshold_accumulated[model] += accuracy_within_threshold(depth_targets, depth_preds[model],
                                                                                    threshold=1.25)
            accuracy_with_threshold2_accumulated[model] += accuracy_within_threshold(depth_targets, depth_preds[model],
                                                                                    threshold=1.25**2)
            accuracy_with_threshold3_accumulated[model] += accuracy_within_threshold(depth_targets, depth_preds[model],
                                                                                    threshold=1.25**3)
            rmse_accumulated[model] += rmse(depth_targets, depth_preds[model])

            #img = depth_preds[model].numpy()[0].transpose(1, 2, 0)
            #plot_image(img, title=model)

    n_batches = np.ceil(len(targets) / batch_size)

    # calculate average over batches, semantic segmentation
    mean_intersection_over_union_avg = {}
    for model in segmentation_models:
        model_name = model[0]
        mean_intersection_over_union_avg[model_name] = mean_intersection_over_union_accumulated[model_name] / n_batches

        print("Model:", model_name, "has jaccard index avg:", mean_intersection_over_union_avg[model_name])

    # calculate average over batches, depth estimation
    accuracy_within_threshold_avg = {}
    accuracy_within_threshold2_avg = {}
    accuracy_within_threshold3_avg = {}
    rmse_avg = {}
    for model in depth_models:
        model_name = model[0]
        accuracy_within_threshold_avg[model_name] = accuracy_with_threshold_accumulated[model_name] / n_batches
        accuracy_within_threshold2_avg[model_name] = accuracy_with_threshold2_accumulated[model_name] / n_batches
        accuracy_within_threshold3_avg[model_name] = accuracy_with_threshold3_accumulated[model_name] / n_batches
        rmse_avg[model_name] = rmse_accumulated[model_name] / n_batches
        print("---")
        print("Model:", model_name, "has accuracy within threshold avg:", accuracy_within_threshold_avg[model_name])
        print("Model:", model_name, "has accuracy within threshold2 avg:", accuracy_within_threshold2_avg[model_name])
        print("Model:", model_name, "has accuracy within threshold3 avg:", accuracy_within_threshold3_avg[model_name])
        print("Model:", model_name, "has rmse avg:", rmse_avg[model_name])
        print("---")


    # TODO speed measurement - dette blir nesten en egen greie
    # TODO lag nice output format


if __name__ == "__main__":

    # location of where to find training, test1, test2
    data_folder = Path("data/perception/test1")
    predictions_folder = Path("data/perception/predictions")

    # lagres på formatet (Navn, lokasjon)
    segmentation_models = [("train10k-lol?", data_folder / "segmentation"),
                           ("semantic-test", predictions_folder / "semantic_test")]

    # lagres på formatet (Navn, lokasjon, invert_pixels_in_loading)
    # ("test1-depth", data_folder / "depth", False)
    depth_models = [("MiDaS-big-test1", predictions_folder / "midas_big_test1", True),
                    ("MiDaS-small-test1", predictions_folder / "midas_small_test1", True),
                    ("AdaBins-nyu-test1", predictions_folder / "adabins_nyu_test1", False),
                    ("AdaBins-kitti-test1", predictions_folder / "adabins_kitti_test1", False)]

    compare_models(data_folder, segmentation_models, depth_models)

