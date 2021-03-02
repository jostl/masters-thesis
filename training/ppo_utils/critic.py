import torch
import torch.nn as nn

from bird_view.models import common
from perception.perception_model import MobileNetUNet


class CriticNetwork(common.ResnetBase):
    def __init__(self, backbone, device, warp=False, pretrained=False, all_branch=False, input_channel=3,
                 perception_ckpt="", n_semantic_classes=6, **kwargs):
        super().__init__(backbone, pretrained=pretrained, input_channel=input_channel, bias_first=False)
        # TODO: Temporary Critic Network.
        if perception_ckpt:
            self.perception = MobileNetUNet(n_semantic_classes)
            self.perception.load_state_dict(torch.load(perception_ckpt, map_location=device))
            self.perception.set_rgb_decoder(use_rgb_decoder=False)
            # Dont calculate gradients for perception layers
            for param in self.perception.parameters():
                param.requires_grad = False
        else:
            self.perception = None

        self.c = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048
        }[backbone]
        self.warp = warp
        self.rgb_transform = common.NormalizeV2(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.deconv = nn.Sequential(
            nn.BatchNorm2d(self.c + 128),
            nn.ConvTranspose2d(self.c + 128, 256, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(True)
        )

        self.value_pred = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Flatten(),
                nn.Linear(64 * 40 * 96, 100),
                nn.ReLU(),
                nn.BatchNorm1d(100),
                nn.Linear(100, 25),
                nn.ReLU(),
                nn.BatchNorm1d(25),
                nn.Linear(25, 1)
            ) for _ in range(4)
        ])

        self.all_branch = all_branch

    def forward(self, state, velocity, command):
        if state.shape[1] == 3:
            state = self.rgb_transform(state)
        if self.perception is not None:
            state = self.perception(state)

        h = self.conv(state)
        b, c, kh, kw = h.size()

        # Late fusion for velocity
        velocity = velocity[..., None, None, None].repeat((1, 128, kh, kw))

        h = torch.cat((h, velocity), dim=1)
        h = self.deconv(h)

        value_pred = [value_pred(h) for value_pred in self.value_pred]
        value_preds = torch.stack(value_pred, dim=1)
        location_pred = common.select_branch(value_preds, command)

        if self.all_branch:
            return location_pred, value_preds

        return location_pred
