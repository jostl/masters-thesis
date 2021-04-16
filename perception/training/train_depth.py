import copy
import math
import time
from pathlib import Path
import random

from kornia.filters import spatial_gradient
from pytorch_msssim import ssim, SSIM
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from perception.custom_datasets import DepthDataset

from perception.training.models import createUNet, createUNetResNet
from perception.utils.visualization import display_images_horizontally

#ssim_loss = SSIM(channel=1, nonnegative_ssim=True)


def depth_loss_function(y_pred, y_true, theta=0.1, maxDepthVal=1):
    # from https://github.com/ialhashim/DenseDepth/blob/ed044069eb99fa06dd4af415d862b3b5cbfab283/loss.py

    # Point-wise depth
    l_depth = torch.mean(torch.abs(y_pred - y_true), dim=-1)

    # Edges
    d_true = spatial_gradient(y_true)
    dx_true = d_true[:, :, 0, :, :]
    dy_true = d_true[:, :, 1, :, :]

    d_pred = spatial_gradient(y_pred)
    dx_pred = d_pred[:, :, 0, :, :]
    dy_pred = d_pred[:, :, 1, :, :]

    l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), dim=-1)

    # Structural similarity (SSIM) index
    l_ssim = torch.clip((1 - ssim(y_true, y_pred, maxDepthVal, nonnegative_ssim=True)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * torch.mean(l_edges)) + (w3 * torch.mean(l_depth))


def create_dataloaders(path, validation_set_size, batch_size=32, max_n_instances=None, augment_strategy=None,
                       num_workers=0, use_transform=None):

    dataset = DepthDataset(root_folder=path, max_n_instances=max_n_instances,
                               augment_strategy=augment_strategy, use_transform=use_transform)
    train_size = int((1 - validation_set_size) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    dataloaders = {"train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                   "val": DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)}
    return dataloaders


def train_model(model, dataloaders, criterion, optimizer, n_epochs, model_save_path, scheduler=None,
                save_model_weights=True, display_img_after_epoch=0):
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
                rgb_input = data[0].to(device, dtype=torch.float32)  # TODO så stor datatype?
                #rgb_target = data[1].to(device, dtype=torch.float32)
                depth_image = data[2].to(device, dtype=torch.float32)
                rgb_raw = data[3]

                # Find the size of rgb_image
                input_size = rgb_input.size(0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(rgb_input)
                    #depth_image = torch.flatten(depth_image, start_dim=1)
                    # TODO loss
                    loss = criterion(outputs, depth_image)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                if display_images is None:
                    # Save image as a numpy array. Used later for displaying predictions.
                    idx = random.randint(0, input_size-1)
                    display_images = [rgb_raw.cpu().numpy()[idx].transpose(1, 2, 0),
                                      depth_image.cpu().numpy()[idx].reshape(160, 384),
                                      outputs.detach().cpu().numpy()[idx].reshape(160, 384)]

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

    model_name = "depth_unet_resnet_ssim6"
    model_save_path = Path("training_logs/perception") / model_name

    validation_set_size = 0.2
    max_n_instances = None
    batch_size = 42
    augment_strategy = "medium"
    path = "data/perception/test1"

    model_save_path.mkdir(parents=True)
    dataloaders = create_dataloaders(path=path, validation_set_size=validation_set_size,
                                                             batch_size=batch_size, max_n_instances=max_n_instances,
                                                             augment_strategy=augment_strategy, num_workers=0,
                                                             use_transform=None)  # "midas_large"

    save_model_weights = True
    display_img_after_epoch = True
    n_epochs = 20

    #model = createDeepLabv3(outputchannels=len(DEFAULT_CLASSES) + 1, backbone=backbone, pretrained=True)
    #model = createUNet()
    model = createUNetResNet()
    #criterion = torch.nn.MSELoss()  # TODO loss
    criterion = depth_loss_function
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer,
                n_epochs=n_epochs, model_save_path=model_save_path, save_model_weights=save_model_weights,
                display_img_after_epoch=display_img_after_epoch)


if __name__ == '__main__':
    main()
