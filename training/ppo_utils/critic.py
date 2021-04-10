import torch
import torch.nn as nn
import torchvision.transforms as transforms

from bird_view.models import common
from bird_view.models.birdview import spatial_softmax_base

STEPS = 5
SPEED_STEPS = 3
COMMANDS = 4
DT = 0.1
CROP_SIZE = 192
PIXELS_PER_METER = 5


class BirdViewCritic(common.ResnetBase):
    def __init__(self, device, backbone='resnet18', input_channel=7, n_step=5, all_branch=False, **kwargs):
        super().__init__(backbone=backbone, input_channel=input_channel, bias_first=False)

        self.deconv = spatial_softmax_base()
        self.value_pred = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Flatten(),
                nn.Linear(64 * 48 * 48, 100),
                nn.ReLU(),
                nn.Linear(100, 25),
                nn.ReLU(),
                nn.Linear(25, 1)
            ) for _ in range(4)
        ])

        self.all_branch = all_branch
        self.transform = transforms.ToTensor()
        self.one_hot = torch.FloatTensor(torch.eye(4))
        self.device = device

    def forward(self, bird_view, velocity, command):
        h = self.conv(bird_view)
        b, c, kh, kw = h.size()

        # Late fusion for velocity
        velocity = velocity[..., None, None, None].repeat((1, 128, kh, kw))

        h = torch.cat((h, velocity), dim=1)
        h = self.deconv(h)

        value_preds = [value_pred(h) for value_pred in self.value_pred]
        value_preds = torch.stack(value_preds, dim=1)

        value_pred = common.select_branch(value_preds, command)

        if self.all_branch:
            return value_pred, value_preds

        return value_pred

    def evaluate(self, birdview, speed, command):
        if self.critic.all_branch:
            state_value, _ = self.critic(birdview, speed, command)
        else:
            state_value = self.critic(birdview, speed, command)
        return state_value.squeeze()

    def prepare_for_eval(self, birdview, speed, command):
        _birdview = self.transform(birdview).to(self.device).unsqueeze(0)
        _speed = torch.FloatTensor([speed]).to(self.device)
        _command = self.one_hot[int(command) - 1].to(self.device).unsqueeze(0)
        return _birdview, _speed, _command
