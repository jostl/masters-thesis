import copy
import math
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from perception.custom_datasets import MultiTaskDataset
from perception.multi_task_criterion import MultiTaskCriterion
from perception.perception_model import MobileNetUNet
from perception.utils.visualization import show_predictions


def create_dataloaders_with_multi_task_dataset(path, validation_set_size, batch_size=32,
                                               n_semantic_classes=6, max_n_instances=-1, augment_strategy=None):
    dataset = MultiTaskDataset(root_folder=path, n_semantic_classes=n_semantic_classes, max_n_instances=max_n_instances,
                               augment_strategy=augment_strategy)
    train_size = int((1 - validation_set_size) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    dataloaders = {"train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                   "val": DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)}
    return dataloaders


def train_model(model, dataloaders, criterion, optimizer, n_epochs, model_save_path, scheduler=None,
                save_model_weights=True, n_displays_per_epoch=0):
    # determine the computational device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)

    best_val_loss = math.inf
    best_model_weights = copy.deepcopy(model.state_dict())

    start = time.time()
    last_time = start

    for epoch in range(n_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))
        display_images = None
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for i, data in tqdm(enumerate(dataloaders[phase])):
                # Get the inputs; data is a list of (RGB, semantic segmentation, depth maps).
                rgb_input = data[0].to(device, dtype=torch.float32)
                rgb_target = data[1].to(device, dtype=torch.float32)
                semantic_image = data[2].to(device, dtype=torch.float32)
                depth_image = data[3].to(device, dtype=torch.float32)
                # Find the size of rgb_image
                input_size = rgb_input.size(0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(rgb_input)

                    loss = criterion(outputs, (rgb_target, semantic_image, depth_image))

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                if phase == "val" and n_displays_per_epoch and display_images is None:
                    # Save image from validation set as a numpy array. Used later for displaying predictions.
                    display_images = (rgb_input, rgb_target, semantic_image, depth_image)

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

        now = time.time()
        time_elapsed = now - last_time
        print("Epoch completed in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60))
        last_time = now

        if n_displays_per_epoch:
            # Show image from the validation set together with decoded predictions
            show_predictions(model, display_images, device, n_displays=n_displays_per_epoch,
                             title="Results from epoch {}".format(epoch + 1))

        # Save the model
        if save_model_weights:
            print("Saving weights to:", model_save_path)
            torch.save(model.state_dict(), model_save_path + "epoch" + str(epoch + 1) + ".pt")

    time_elapsed = time.time() - start
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))

    # load the best weights
    model.load_state_dict(best_model_weights)

    # Save the model
    if save_model_weights:
        print("Saving best weights to:", model_save_path)
        torch.save(model.state_dict(), model_save_path)
    return model


def main():
    model_name = "perception_test"
    model_save_path = "training_logs/perception/{}".format(model_name)
    validation_set_size = 0.2
    max_n_instances = -1
    batch_size = 2
    n_semantic_classes = 6
    augment_strategy = "super_hard"
    path = "data/perception/carla_test3"
    dataloaders = create_dataloaders_with_multi_task_dataset(path=path, validation_set_size=validation_set_size,
                                                             n_semantic_classes=n_semantic_classes,
                                                             batch_size=batch_size, max_n_instances=max_n_instances,
                                                             augment_strategy=augment_strategy)

    save_model_weights = True
    n_displays_per_epoch = 3
    n_epochs = 30

    model = MobileNetUNet(n_semantic_classes=n_semantic_classes)
    criterion = MultiTaskCriterion()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
                n_epochs=n_epochs, model_save_path=model_save_path, save_model_weights=save_model_weights,
                n_displays_per_epoch=n_displays_per_epoch)

if __name__ == '__main__':
    main()
