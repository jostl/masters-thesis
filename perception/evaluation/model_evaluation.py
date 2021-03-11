from torch.utils.data import DataLoader
from pathlib import Path

from perception.custom_datasets import ComparisonDataset
from perception.utils.visualization import display_originals_with_decoded, display_batch_comparison


def compare_models(data_folder, segmentation_models, depth_models):
    targets = ComparisonDataset(data_folder, segmentation_models, depth_models, n_semantic_classes=6,
                                max_n_instances=7)

    dataloader = DataLoader(targets, batch_size=5, shuffle=False, num_workers=0,
                            pin_memory=True)

    for rgb_target, segmentation_target, depth_target, segmentation_preds, depth_preds in dataloader:
        pass


if __name__ == "__main__":

    # location of where to find training, test1, test2
    data_folder = Path("data/perception/train_10k/train")
    predictions_folder = Path("data/perception/predictions")

    # lagres på formatet (Navn, lokasjon)
    segmentation_models = [("nvidia-test", predictions_folder / "nvidia_test")]
    depth_models = [("MiDaS-test", predictions_folder / "midas_test")]

    compare_models(data_folder, segmentation_models, depth_models)

