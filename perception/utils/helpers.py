# Original file:
# https://github.com/AudunWA/master-thesis-code/blob/master/prediction_pipeline/utils/helpers.py
# Modified by Martin Hermansen and Jostein Lilleløkken to fit our needs for our specialization project.
# Many of the changes are inspired by and/or based on the work by:
# Arbo&Dalen (2020).


import os
import time
from multiprocessing.dummy import Pool
from pathlib import Path

import cv2
import numpy as np


def convert_segmentation_images(input_folder, output_folder, classes_array):
    filenames = os.listdir(input_folder)

    verify_folder_exists(output_folder)

    i = 0
    for filename in filenames:

        # In BGR format
        img = cv2.cvtColor(cv2.imread(input_folder + "/" + filename), cv2.COLOR_BGR2RGB)
        output_img = np.zeros(img.shape)

        for j, class_colors in enumerate(classes_array):
            for color in class_colors:
                mask = (img == color).all(axis=2)
                output_img[mask] = [j + 1, 0, 0]

        write_path = str(output_folder + "/" + filename)

        cv2.imwrite(write_path, output_img)

        i += 1
        if i % 100 == 0:
            print("Progress: ", i, " of ", len(filenames))


def read_and_crop_image(input_dir, output_dir, filename, counter):
    try:
        file_path = input_dir + "/" + filename
        write_path = os.path.join(output_dir, filename)

        if ".npz" in filename:
            try:
                img = np.load(file_path)["arr_0"]
                img = img.reshape((img.shape[0], img.shape[1], 1)) * 255 * 3  # 1 Channel
                img = img.astype("uint8")
            except Exception as e:
                print("Exception:", e)
                raise e
        else:
            img = cv2.imread(file_path)
        try:
            cv2.imwrite(write_path.replace("npz", "jpg"), crop_and_resize_img(img))
        except Exception as e:
            print("Exception on write:", e)
            raise e
        if counter % 500 == 0:
            print("Progress:", counter)
    except Exception as e:
        print("Exception:", e)


def crop_and_resize(input_dir, output_dir):
    verify_folder_exists(Path(output_dir))
    filenames = list(os.listdir(input_dir))
    print("output_dir", output_dir)

    print("Processing {} images".format(len(filenames)))
    with Pool(processes=8) as pool:  # this should be the same as your processor cores (or less)
        chunksize = 56  # making this larger might improve speed (less important the longer a single function call takes)
        print("chunksize", chunksize)

        result = pool.starmap_async(read_and_crop_image,  # function to send to the worker pool
                                    ((input_dir, output_dir, file, i) for i, file in enumerate(filenames)),
                                    # generator to fill in function args
                                    chunksize)  # how many jobs to submit to each worker at once
        while not result.ready():  # print out progress to indicate program is still working.
            # with counter.get_lock(): #you could lock here but you're not modifying the value, so nothing bad will happen if a write occurs simultaneously
            # just don't `time.sleep()` while you're holding the lock

            time.sleep(.1)
        print('\nCompleted all images')


def crop_and_resize_img(img):
    side_len = min(img.shape[0], img.shape[1])
    side_len -= side_len % 32
    cropped_img = img[0:side_len, 0:side_len]
    return cv2.resize(cropped_img, (288, 256), interpolation=cv2.INTER_NEAREST)


def verify_folder_exists(path):
    if not os.path.exists(str(path)):
        os.makedirs(str(path))


def get_segmentation_tensor(image: np.ndarray, classes):
    n_classes = len(classes) + 1
    height, width, _ = image.shape
    segmentation_labels = np.zeros((height, width, n_classes))
    image = image[:, :, 0]
    for i, c in enumerate(classes):
        segmentation_labels[:, :, i] = (image == c).real
        filter = (image == c).real
        segmentation_labels[:, :, -1] = np.logical_or(segmentation_labels[:, :, -1], filter)
    segmentation_labels[:, :, -1] = np.logical_not(segmentation_labels[:, :, -1])
    return segmentation_labels


if __name__ == '__main__':
    semantic_image = "data/perception/test1/segmentation/clear_noon_1228_163.png"
    rgb_image = "data/perception/test1/rgb/clear_noon_1228_163.png"
    rgb = cv2.cvtColor(cv2.imread(rgb_image), cv2.COLOR_BGR2RGB)

    classes = [8, 7, 6, 4, 10, 5, 18, 14]
    semantic_img = get_segmentation_tensor(cv2.cvtColor(cv2.imread(semantic_image), cv2.COLOR_BGR2RGB),
                                           classes=classes)

    import matplotlib.pyplot as plt
    from perception.utils.visualization import get_rgb_segmentation, get_segmentation_colors

    # plot_segmentation(semantic_img)
    colors = get_segmentation_colors(len(classes) + 1, class_indxs=classes)
    rgb_segmentation = get_rgb_segmentation(semantic_img, colors) / 255
    plt.imshow(rgb_segmentation)
    plt.show()
    plt.imshow(rgb)
    plt.show()
