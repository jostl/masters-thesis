import copy
import math
import time
from pathlib import Path
import random

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from perception.custom_datasets import SegmentationDataset
from perception.deeplabv3.models import createDeepLabv3, createFCN
from perception.utils.segmentation_labels import DEFAULT_CLASSES
from perception.utils.visualization import display_images_horizontally, get_rgb_segmentation, get_segmentation_colors


def create_dataloaders(path, validation_set_size, batch_size=32, semantic_classes=DEFAULT_CLASSES,
                       max_n_instances=None, augment_strategy=None, num_workers=0):

    dataset = SegmentationDataset(root_folder=path, semantic_classes=semantic_classes, max_n_instances=max_n_instances,
                               augment_strategy=augment_strategy)
    train_size = int((1 - validation_set_size) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    dataloaders = {"train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                   "val": DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)}
    return dataloaders


def train_model(model, dataloaders, criterion, optimizer, n_epochs, model_save_path, scheduler=None,
                save_model_weights=True, display_img_after_epoch=0, semantic_classes=DEFAULT_CLASSES):
    # determine the computational device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)

    best_val_loss = math.inf
    best_model_weights = copy.deepcopy(model.state_dict())

    start = time.time()
    last_time = start

    # Tensorboard logging
    train_log_path = model_save_path / "logs/train"
    val_log_path = model_save_path / "logs/val"
    train_log_path.mkdir(parents=True)
    val_log_path.mkdir(parents=True)
    writer_train = SummaryWriter(model_save_path / "logs/train")
    writer_val = SummaryWriter(model_save_path / "logs/val")

    for epoch in range(n_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            display_images = None

            for i, data in tqdm(enumerate(dataloaders[phase])):
                # Get the inputs; data is a list of (RGB, semantic segmentation, depth maps).
                rgb_input = data[0].to(device, dtype=torch.float32)
                #rgb_target = data[1].to(device, dtype=torch.float32)
                semantic_image = data[2].to(device, dtype=torch.float32)
                rgb_raw = data[3]

                # Find the size of rgb_image
                input_size = rgb_input.size(0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(rgb_input)
                    target_classes = semantic_image.argmax(dim=1)

                    loss = criterion(outputs["out"], target_classes)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                if display_images is None:
                    # Save image as a numpy array. Used later for displaying predictions.
                    idx = random.randint(0, input_size-1)
                    display_images = [rgb_raw.cpu().numpy()[idx].transpose(1, 2, 0),
                                      semantic_image.cpu().numpy()[idx].transpose(1, 2, 0),
                                      outputs["out"].detach().cpu().numpy()[idx].transpose(1, 2, 0)]

                # statistics
                running_loss += loss.item() * input_size

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            dataset_size = len(dataloaders["train"].dataset) if phase == "train" else len(dataloaders["val"].dataset)
            epoch_loss = running_loss / dataset_size

            print('{} Loss: {:.6f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == "val" and epoch_loss < best_val_loss:
                print("val loss record low, saving these weights...")
                best_val_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())

            writer = writer_train if phase == "train" else writer_val
            writer.add_scalar("epoch_loss", epoch_loss, epoch)

            # Show predicted image together with decoded predictions
            class_colors = get_segmentation_colors(len(semantic_classes) + 1, class_indxs=semantic_classes)
            semantic_target_rgb = get_rgb_segmentation(semantic_image=display_images[1],
                                                       class_colors=class_colors)
            semantic_pred_rgb = get_rgb_segmentation(semantic_image=display_images[2],
                                                     class_colors=class_colors)
            semantic_target_rgb = semantic_target_rgb / 255
            display_images[1] = semantic_target_rgb
            semantic_pred_rgb = semantic_pred_rgb / 255
            display_images[2] = semantic_pred_rgb

            display = True if phase == "val" and display_img_after_epoch else False

            figtitle = "{} visualization after epoch {}".format(phase, epoch)
            subtitles = ["augmented input", "ground truth", "prediction"]

            img = display_images_horizontally(display_images, fig_width=10, fig_height=2, display=display,
                                              title=figtitle, subplot_titles=subtitles)

            writer.add_image("{} comparison".format(phase), img.transpose(2, 0, 1), epoch)

        now = time.time()
        time_elapsed = now - last_time
        print("Epoch completed in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60))
        last_time = now

        # Save the model
        if save_model_weights:
            path = model_save_path / "epoch-{}.pt".format(epoch+1)
            print("Saving weights to:", path)
            torch.save(model.state_dict(), path)

    time_elapsed = time.time() - start
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))

    # load the best weights
    model.load_state_dict(best_model_weights)

    # Save the model
    if save_model_weights:
        path = model_save_path / "best_weights.pt"
        print("Saving best weights to:", path)
        torch.save(model.state_dict(), path)
    return model


def main():

    model_name = "test_batch_aug_rng9"
    model_save_path = Path("training_logs/perception") / model_name

    validation_set_size = 0.2
    max_n_instances = 10
    batch_size = 2
    semantic_classes = DEFAULT_CLASSES
    augment_strategy = "super_hard"
    path = "data/perception/test1"

    model_save_path.mkdir(parents=True)
    dataloaders = create_dataloaders(path=path, validation_set_size=validation_set_size,
                                                             semantic_classes=semantic_classes,
                                                             batch_size=batch_size, max_n_instances=max_n_instances,
                                                             augment_strategy=augment_strategy, num_workers=4)

    save_model_weights = True
    display_img_after_epoch = False
    n_epochs = 3

    use_class_weights = False
    # weight any class how you'd like here - we dont need to normalize these as long as reduction="mean" in criterion
    weights = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
    backbone = "resnet50"

    #model = createDeepLabv3(outputchannels=len(DEFAULT_CLASSES) + 1, backbone=backbone, pretrained=True)
    model = createDeepLabv3(outputchannels=len(DEFAULT_CLASSES) + 1, backbone=backbone, pretrained=True)
    criterion = torch.nn.CrossEntropyLoss(weight=weights if use_class_weights else None)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
                n_epochs=n_epochs, model_save_path=model_save_path, save_model_weights=save_model_weights,
                display_img_after_epoch=display_img_after_epoch, semantic_classes=semantic_classes)


if __name__ == '__main__':
    main()
