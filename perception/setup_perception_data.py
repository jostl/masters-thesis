# Original file:
# https://github.com/AudunWA/master-thesis-code/blob/master/prediction_pipeline/generate_segmentation_files.py
# Modified by Martin Hermansen and Jostein Lilleløkken to fit our needs for our specialization project
# Many of the changes are inspired by and/or based on the work by
# Arbo&Dalen (2020)

# This file is for preparing (i.e. cropping and resizing) RGB and SEMANTIC SEGMENTATION images.
# For semantic segmentation images, it will store it in a practical way which lends itself to be easily read and
# used for training in a neural network.
# In "main()", change path strings for all datasets to correct location.

import os
from pathlib import Path
from shutil import copyfile

import cv2
from tqdm import tqdm

from perception.utils.helpers import crop_and_resize, convert_segmentation_images
from perception.utils.segmentation_labels import CARLA_CLASSES


def prepare_for_multi_task_dataset(folder_name, output_folder):
    # If you want to use MultiTaskDataset, run this method after setting up data
    for dataset_directory in list(os.listdir(folder_name)):
        for sensor_directory in list(os.listdir(folder_name + "/" + dataset_directory)):
            if sensor_directory != "test":
                print("Copying {} files from {} to {}".format(sensor_directory, dataset_directory,
                                                              output_folder + "/" + sensor_directory))
                for filename in tqdm(list(os.listdir(folder_name + "/" + dataset_directory + "/" + sensor_directory))):
                    copyfile(folder_name + "/" + dataset_directory + "/" + sensor_directory + "/" + filename,
                             output_folder + "/" + sensor_directory + "/" + filename)


def remove_bad_segmentation_images(folder_path, output_folder, threshold):
    filenames = list(os.listdir(folder_path + "/segmentation"))
    remove_count = 0
    sensors = ["rgb", "segmentation"]
    for filename in filenames:
        image = cv2.imread(folder_path + "/segmentation/" + filename)
        height, width, _ = image.shape
        total_pixels = height * width
        unlabeled_class_count = (image[:, :, 0] == 0).sum(axis=1).sum()
        unlabeled_fraction = unlabeled_class_count / total_pixels
        if unlabeled_fraction <= threshold:
            for sensor in sensors:
                if sensor == "depth":
                    filename = filename.split(".")[0] + "_disp.jpeg"
                copyfile(folder_path + "/" + sensor + "/" + filename,
                         output_folder + "/" + sensor + "/" + filename)
        else:
            remove_count += 1

    print(len(filenames))
    print(remove_count)


def main(carla_path):
    # TODO: This code is the most ugly code ever written, need to clean it up later
    # TODO: Cleaning up betyr blant annet å bruke pathlib og ikke strings
    if carla_path:
        print("Preparing CARLA semantic segmentation labels")
        convert_segmentation_images(carla_path + "/segmentation", carla_path + "/segmentation",
                                        CARLA_CLASSES)


if __name__ == '__main__':
    carla_path = "data/perception/carla_test3"
    main(carla_path=carla_path)

    # prepare_for_multi_task_dataset("data/perception/prepped_256x288", "data/perception/prepped_256x288_mtl")

    carla_path = "data/perception/dry_weathers/carla"
    carla_cleaned_path = "data/perception/dry_weathers/carla_clean"
    # remove_bad_segmentation_images(carla_path, carla_cleaned_path, threshold=0.9)
