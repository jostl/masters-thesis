from pathlib import Path

import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from perception.utils.helpers import get_segmentation_array


class MultiTaskDataset(Dataset):
    """Dataset of folder with rgb, segmentation and depth subfolders"""

    def __init__(self, root_folder: str, n_semantic_classes, transform=None, max_n_instances=-1):
        self.root_folder = Path(root_folder)
        self.transform = transform
        self.n_semantic_classes = n_semantic_classes

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

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        def transpose(img, normalize: bool):
            img = img.transpose(2, 0, 1)
            return img / 255 if normalize else img

        rgb_img = transpose(cv2.imread(str(self.rgb_imgs[idx])), normalize=True)
        semantic_img = transpose(
            get_segmentation_array(cv2.imread(str(self.semantic_imgs[idx])), n_classes=self.n_semantic_classes),
            normalize=False)
        depth_img =np.array([cv2.imread(str(self.depth_imgs[idx]), cv2.IMREAD_GRAYSCALE)])/255

        return rgb_img, semantic_img, depth_img


if __name__ == '__main__':
    dataset = MultiTaskDataset("data/perception/prepped_256x288_mtl", n_semantic_classes=6)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                            pin_memory=True)

    for i, data in enumerate(dataloader):
        rgb, semantic, depth = data
        print(semantic.shape)
