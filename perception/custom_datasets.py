from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

import augmenter
from perception.utils.helpers import get_segmentation_tensor
from perception.utils.segmentation_labels import DEFAULT_CLASSES


class MultiTaskDataset(Dataset):
    """Dataset of folder with rgb, segmentation and depth subfolders"""

    def __init__(self, root_folder: str, transform=None, semantic_classes=DEFAULT_CLASSES, max_n_instances=-1, augment_strategy=None):
        self.root_folder = Path(root_folder)
        self.transform = transform
        self.semantic_classes = semantic_classes

        self.rgb_folder = self.root_folder / "rgb"
        self.semantic_folder = self.root_folder / "segmentation"
        self.depth_folder = self.root_folder / "depth"

        self.rgb_imgs = [x for x in self.rgb_folder.iterdir()]
        self.semantic_imgs = [x for x in self.semantic_folder.iterdir()]
        self.depth_imgs = [x for x in self.depth_folder.iterdir()]

        self.rgb_imgs.sort()
        self.semantic_imgs.sort()
        self.depth_imgs.sort()

        self.rgb_imgs = self.rgb_imgs[:max_n_instances]
        self.semantic_imgs = self.semantic_imgs[:max_n_instances]
        self.depth_imgs = self.depth_imgs[:max_n_instances]

        assert len(self.rgb_imgs) == len(self.depth_imgs)
        assert len(self.rgb_imgs) == len(self.semantic_imgs)

        self.num_imgs = len(self.rgb_imgs)

        print("Len of dataset is:", self.num_imgs)
        print("augment with", augment_strategy)
        if augment_strategy is not None and augment_strategy != None:
            self.augmenter = getattr(augmenter, augment_strategy)
        else:
            self.augmenter = None
        self.batch_read_number = 819200

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        def transpose(img, normalize: bool):
            img = img.transpose(2, 0, 1)
            return img / 255 if normalize else img

        def read_rgb(img_path):
            return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        rgb_target = read_rgb(str(self.rgb_imgs[idx]))
        if self.augmenter:
            rgb_input = self.augmenter(self.batch_read_number).augment_image(rgb_target)
        else:
            rgb_input = rgb_target
        rgb_input = transpose(rgb_input, normalize=True)
        rgb_target = transpose(rgb_target, normalize=True)
        semantic_img = transpose(
            get_segmentation_tensor(cv2.imread(str(self.semantic_imgs[idx])), classes=self.semantic_classes),
            normalize=False)
        depth_img = np.array([cv2.imread(str(self.depth_imgs[idx]), cv2.IMREAD_GRAYSCALE)]) / 255
        self.batch_read_number += 1
        return rgb_input, rgb_target, semantic_img, depth_img


if __name__ == '__main__':
    dataset = MultiTaskDataset("data/perception/prepped_256x288_mtl", semantic_classes=DEFAULT_CLASSES)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                            pin_memory=True)

    for i, data in enumerate(dataloader):
        rgb, semantic, depth = data
        print(semantic.shape)
