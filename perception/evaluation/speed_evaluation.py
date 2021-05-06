import time

import numpy as np
import torch
from tqdm import tqdm

from perception.training.models import createDeepLabv3, createFCN, createUNetResNet, createUNet, createUNetResNetSemSeg, createMidas
from perception.training.train_semseg import create_dataloaders as create_semseg_dataloaders
from perception.training.train_depth import create_dataloaders as create_depth_dataloaders
from perception.utils.segmentation_labels import DEFAULT_CLASSES


def time_predictions(model: torch.nn.Module, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available()

    model.to(device)
    model.eval()

    times = []
    for i, data in tqdm(enumerate(dataloader)):
        # Get the inputs; data is a list of (RGB, semantic segmentation, depth maps).
        rgb_input = data[0].to(device, dtype=torch.float32)
        # rgb_target = data[1].to(device, dtype=torch.float32)
        #semantic_image = data[2].to(device, dtype=torch.float32)
        #rgb_raw = data[3]

        with torch.set_grad_enabled(False):
            t0 = time.time()
            outputs = model(rgb_input)
            torch.cuda.current_stream().synchronize()
            t1 = time.time()
            times.append(t1-t0)
            #print("Time: ", times[i])

    return times


if __name__=="__main__":
    validation_set_size = 0.1
    max_n_instances = 666  # gives 500 measurements
    batch_size = 1
    semantic_classes = DEFAULT_CLASSES
    augment_strategy = "super_hard"
    path = "data/perception/test1"
    #backbone = "mobilenet"
    #model = createFCN(outputchannels=len(DEFAULT_CLASSES) + 1, backbone="resnet101")
    model = createDeepLabv3(outputchannels=len(DEFAULT_CLASSES) + 1, backbone="mobilenet", pretrained=True)
    #model = createUNetResNetSemSeg(n_classes=len(DEFAULT_CLASSES) + 1)
    dataloaders = create_semseg_dataloaders(path=path, validation_set_size=validation_set_size,
                                    semantic_classes=semantic_classes,
                                    batch_size=batch_size, max_n_instances=max_n_instances,
                                    augment_strategy=augment_strategy, num_workers=0)

    #model = createMidas(True)
    #dataloaders = create_depth_dataloaders(path=path, validation_set_size=validation_set_size, batch_size=batch_size,
    #                                       max_n_instances=max_n_instances, augment_strategy=augment_strategy, num_workers=0)

    times = time_predictions(model, dataloaders["train"])

    print("Slowest pred:", np.max(times))
    print("Fastest pred:", np.min(times))
    print("Average prediction time:", np.average(times))
    print("Average stable prediction time:", np.average(times[100:]))

