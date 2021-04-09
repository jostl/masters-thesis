import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
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
        # TODO må kanskje endre head også da

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