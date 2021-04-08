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
    elif backbone == "mobilenet":
        print("DeepLabv3: Using mobilenet as backbone")
        # TODO må kanskje endre head også da
        #raise NotImplementedError
        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)
    else:
        print("DeepLabv3: Using resnet101 as backbone")
        model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained,
                                                        progress=True)

    model.aux_classifier = None
    #for param in model.parameters():
    #    param.requires_grad = False

    model.classifier = DeepLabHead(2048, outputchannels)

    return model


def createFCN(outputchannels=1, backbone="resnet50", pretrained=True):
    if backbone == "resnet50":
        print("FCN: Using resnet50 as backbone")
        model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=True,
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


def createMidas(use_large_model=True):

    if use_large_model:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    else:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

    #return midas

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if use_large_model:
        transform = midas_transforms.default_transform
    else:
        transform = midas_transforms.small_transform
    return midas
    img = cv2.imread(filename)
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


if __name__ == "__main__":
    #createDeepLabv3(outputchannels=9, backbone="resnet50", pretrained=True)
    #createMidas()
    createFCN(outputchannels=9, backbone="resnet101", pretrained=True)