import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision import models


def createDeepLabv3(outputchannels=1, backbone="resnet50", pretrained=True):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """

    if backbone == "resnet50":
        print("DeepLabv3: Using resnet50 as backbone")
        model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained,
                                                       progress=True)
        model.classifier = DeepLabHead(2048, outputchannels)

    elif backbone == "mobilenet":
        print("DeepLabv3: Using mobilenet as backbone")
        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=pretrained, progress=True)
        model.classifier = DeepLabHead(960, outputchannels)
    else:
        print("DeepLabv3: Using resnet101 as backbone")
        model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained,
                                                        progress=True)
        model.classifier = DeepLabHead(2048, outputchannels)

    model.aux_classifier = None
    #for param in model.parameters():
    #    param.requires_grad = False
    return model


def createFCN(outputchannels=1, backbone="resnet50", pretrained=True):
    if backbone == "resnet50":
        print("FCN: Using resnet50 as backbone")
        model = models.segmentation.fcn_resnet50(pretrained=pretrained, progress=True,
                                                       num_classes=21, aux_loss=False)
    else:
        print("FCN: Using resnet101 as backbone")
        model = models.segmentation.fcn_resnet101(pretrained=pretrained, progress=True,
                                                  num_classes=21, aux_loss=False)

    model.aux_classifier = None
    #for param in model.parameters():
    #    param.requires_grad = False

    model.classifier = FCNHead(2048, outputchannels)

    return model

def createUNet():
    from perception.unet.unet_model import UNet
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    return model

def createUNetResNet():
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1,
                     activation="sigmoid")
    return model


def createUNetResNetSemSeg(n_classes):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=n_classes,
                     activation="softmax2d")
    return model


def createMidas(use_large_model=True):

    if use_large_model:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    else:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

    return midas
    """
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if use_large_model:
        transform = midas_transforms.default_transform
    else:
        transform = midas_transforms.small_transform

    import cv2
    import urllib.request
    import numpy as np
    import matplotlib.pyplot as plt

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    img = cv2.imread("data/perception/test1/rgb/clear_noon_1823_463.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    plt.imshow(output)
    plt.show()
    """


if __name__ == "__main__":
    #createDeepLabv3(outputchannels=9, backbone="resnet50", pretrained=True)
    createMidas()
    #createFCN(outputchannels=9, backbone="resnet101", pretrained=True)
