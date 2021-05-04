from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from torchvision.utils import save_image
import numpy as np

from perception.custom_datasets import ComparisonDataset, DepthDataset, SegmentationDataset
from perception.training.models import createUNet, createFCN, createDeepLabv3, createUNetResNet, createUNetResNetSemSeg
from perception.utils.visualization import plot_image, get_segmentation_colors, get_rgb_segmentation, \
    display_images_horizontally, plot_segmentation
from perception.utils.segmentation_labels import DEFAULT_CLASSES, CARLA_CLASSES


def semseg_rgb_to_red_channel_bgr(input_img, n_classes=len(DEFAULT_CLASSES)+1):
    # takes an img with numpy shape, HxWxC
    # outputs bgr format for opencv imwrite

    height, width = input_img.shape[0:2]
    output_img = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    pixel_classes = np.argmax(input_img, axis=2)
    expanded_pixel_classes = np.expand_dims(pixel_classes, axis=2)

    output_img[:, :] = expanded_pixel_classes
    output_img[:, :, :2] = 0

    convert = lambda pixel: DEFAULT_CLASSES[pixel] if pixel < len(DEFAULT_CLASSES) else 0
    for i in range(len(output_img)):
        for j in range(len(output_img[0])):
            output_img[i, j, 2] = convert(output_img[i, j, 2])

    return output_img

def main():
    test = "test1"
    data_folder = Path("data/perception") / test
    save_folder = Path("data/perception/predictions/depth/unet_resnet34") / (test)

    save_folder.mkdir(parents=True, exist_ok=False)
    targets = DepthDataset(data_folder, max_n_instances=None)
    #targets = SegmentationDataset(data_folder, semantic_classes=DEFAULT_CLASSES, max_n_instances=None)

    dataloader = DataLoader(targets, batch_size=1, shuffle=False, num_workers=0)

    model = createUNetResNet()
    #model = createUNetResNetSemSeg(n_classes=(len(DEFAULT_CLASSES)+1))
    #model = createDeepLabv3(outputchannels=len(DEFAULT_CLASSES)+1, backbone="resnet101")
    #model = createFCN(outputchannels=len(DEFAULT_CLASSES)+1, backbone="resnet101")
    #model_weights = Path("models/perception/unet_best_weights.pt")
    model_weights = Path("models/perception/unet_resnet34_pretrained_best_weights.pt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available()
    model.to(device)
    model.load_state_dict(torch.load(model_weights))
    model.eval()

    for i, data in tqdm(enumerate(dataloader)):
        # Get the inputs; data is a list of (RGB, semantic segmentation, depth maps).
        rgb_input = data[0].to(device, dtype=torch.float32)
        #depth_target = data[2].to(device, dtype=torch.float32)
        semantic_target = data[2].to(device, dtype=torch.float32)
        rgb_raw = data[3]
        img_name = data[4][0].split("\\")[-1]

        with torch.set_grad_enabled(False):
            outputs = model(rgb_input)

        img_save_path = str((save_folder / img_name).absolute())
        #image_gray = cv2.cvtColor(outputs[0].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)

        if True:
            #print("Saving depth prediction to:", img_save_path)
            #cv2.imwrite(img_save_path, outputs[0].cpu().byte().numpy().transpose(1, 2, 0))
            save_image(outputs[0], img_save_path)


        if False:
            plot_image(rgb_raw[0].numpy().transpose(1, 2, 0), title="rgb raw, "+img_name)
            plot_image(rgb_input[0].cpu().numpy().transpose(1, 2, 0), title="rgb input, "+img_name)
            plot_image(depth_target[0].cpu().numpy().transpose(1, 2, 0), title="ground truth, "+img_name, cmap="gray")
            #plot_image(semantic_target[0].cpu().numpy().transpose(1, 2, 0), title="ground truth, " + img_name, cmap="gray")
            plot_image(outputs[0].cpu().numpy().transpose(1, 2, 0), title="predicted, "+img_name, cmap="gray")
            pass

        if False:
            class_colors = get_segmentation_colors(len(DEFAULT_CLASSES) + 1, class_indxs=DEFAULT_CLASSES)
            semantic_target_rgb = get_rgb_segmentation(semantic_image=semantic_target[0].cpu().numpy().transpose(1, 2, 0),
                                                       class_colors=class_colors)
            semantic_pred_rgb = get_rgb_segmentation(semantic_image=outputs[0].cpu().numpy().transpose(1, 2, 0),
                                                     class_colors=class_colors)
            semantic_target_rgb = semantic_target_rgb / 255
            semantic_pred_rgb = semantic_pred_rgb / 255

            display_images = [rgb_raw[0].numpy().transpose(1, 2, 0), semantic_target_rgb, semantic_pred_rgb]
            subtitles = ["rgb raw", "target", "pred"]

            img = display_images_horizontally(display_images, fig_width=10, fig_height=2, display=True,
                                              title=img_name, subplot_titles=subtitles)

        if False:
            #print("Saving semseg prediction to:", img_save_path)
            red_channel_img = semseg_rgb_to_red_channel_bgr(outputs["out"][0].cpu().numpy().transpose(1, 2, 0))
            cv2.imwrite(img_save_path, red_channel_img)


if __name__=="__main__":
    main()
