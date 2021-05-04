import math

import numpy as np

import torch
import torch.nn as nn

from . import common
from .agent import Agent
from .controller import CustomController, PIDController
from .controller import ls_circle
from perception.utils.segmentation_labels import DEFAULT_CLASSES
from perception.utils.helpers import get_segmentation_tensor
from perception.training.models import createUNet, createUNetResNetSemSeg, createUNetResNet
CROP_SIZE = 192
STEPS = 5
COMMANDS = 4
DT = 0.1
CROP_SIZE = 192
PIXELS_PER_METER = 5

        
class ImagePolicyModelSS(common.ResnetBase):
    def __init__(self, backbone, warp=False, pretrained=False, all_branch=False, input_channel=3, **kwargs):
        super().__init__(backbone, pretrained=pretrained, input_channel=input_channel, bias_first=False)
        
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
            nn.ConvTranspose2d(self.c + 128,256,3,2,1,1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.ReLU(True),
        )
        
        if warp:
            ow,oh = 48,48
        else:
            ow,oh = 96,40 
        
        self.location_pred = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64,STEPS,1,1,0),
                common.SpatialSoftmax(ow,oh,STEPS),
            ) for i in range(4)
        ])
        
        self.all_branch = all_branch

    def forward(self, image, velocity, command):
        if self.warp:
            warped_image = tgm.warp_perspective(image, self.M, dsize=(192, 192))
            resized_image = resize_images(image)
            image = torch.cat([warped_image, resized_image], 1)

        if image.shape[1] == 3:
            image = self.rgb_transform(image)

        h = self.conv(image)
        b, c, kh, kw = h.size()

        # Late fusion for velocity
        velocity = velocity[..., None, None, None].repeat((1, 128, kh, kw))

        h = torch.cat((h, velocity), dim=1)
        h = self.deconv(h)

        location_preds = [location_pred(h) for location_pred in self.location_pred]
        location_preds = torch.stack(location_preds, dim=1)
        location_pred = common.select_branch(location_preds, command)

        if self.all_branch:
            return location_pred, location_preds

        return location_pred


class FullModel(nn.Module):
    def __init__(self, image_backbone, all_branch, image_pretrained=False, image_ckpt=""):
        super(FullModel, self).__init__()
        #self.depth_model = createUNet()
        #self.depth_model.load_state_dict(torch.load("models/perception/depth/unet_best_weights.pt"))'
        self.depth_model = createUNetResNet()
        self.depth_model.load_state_dict(torch.load("models/perception/depth/unet_resnet34_pretrained_best_weights.pt"))
        self.depth_model.eval()

        self.semseg_model = createUNetResNetSemSeg(len(DEFAULT_CLASSES) + 1)
        self.semseg_model.load_state_dict(torch.load("models/perception/segmentation/unet_resnet50_weighted_tlights_5_epoch-29.pt"))
        self.semseg_model.eval()

        self.return_cv_preds = True

        for param in self.depth_model.parameters():
            param.requires_grad = False
        for param in self.semseg_model.parameters():
            param.requires_grad = False

        self.normalize_rgb = common.NormalizeV2(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.image_model = ImagePolicyModelSS(image_backbone,
                                              pretrained=image_pretrained,
                                              all_branch=all_branch, input_channel=len(DEFAULT_CLASSES) + 5)
        if image_ckpt:
            self.image_model.load_state_dict(torch.load(image_ckpt))
        self.all_branch = all_branch

    def forward(self, rgb_raw, velocity, command):

        rgb_norm = self.normalize_rgb(rgb_raw)
        with torch.no_grad():
            dept_pred = torch.clip(self.depth_model(rgb_norm), 0, 1)
            semseg_pred = self.semseg_model(rgb_norm)
        semseg_argmax = torch.argmax(semseg_pred, dim=1)
        semseg_pred = nn.functional.one_hot(semseg_argmax, num_classes=len(DEFAULT_CLASSES) + 1).permute(0, 3, 1, 2)
        _input = torch.cat([rgb_norm, semseg_pred, dept_pred], dim=1)
        if self.return_cv_preds:
            return self.image_model(_input, velocity, command), semseg_pred, dept_pred
        return self.image_model(_input, velocity, command)


class ImageAgent(Agent):
    def __init__(self, steer_points=None, pid=None, gap=5,
                 camera_args={'x': 384, 'h': 160, 'fov': 90, 'world_y': 1.4, 'fixed_offset': 4.0}, use_cv=False, **kwargs):
        super().__init__(**kwargs)

        self.fixed_offset = float(camera_args['fixed_offset'])
        print("Offset: ", self.fixed_offset)
        w = float(camera_args['w'])
        h = float(camera_args['h'])
        self.img_size = np.array([w, h])
        self.gap = gap

        if steer_points is None:
            steer_points = {"1": 4, "2": 3, "3": 2, "4": 2}

        if pid is None:
            # Original LBC pid
            pid = {
                "1": {"Kp": 0.5, "Ki": 0.20, "Kd": 0.0},  # Left
                "2": {"Kp": 0.7, "Ki": 0.10, "Kd": 0.0},  # Right
                "3": {"Kp": 1.0, "Ki": 0.10, "Kd": 0.0},  # Straight
                "4": {"Kp": 1.0, "Ki": 0.50, "Kd": 0.0},  # Follow
            }

            # Reproduction model-10 pid
            #pid = {
            #    "1": {"Kp": 0.85, "Ki": 0.20, "Kd": 0.0},  # Left
            #    "2": {"Kp": 0.6, "Ki": 0.10, "Kd": 0.0},  # Right
            #    "3": {"Kp": 1.0, "Ki": 0.10, "Kd": 0.0},  # Straight
            #    "4": {"Kp": 2.0, "Ki": 0.2, "Kd": 0.1},  # Follow
            #}
            #pid2 = {
            #    "1": {"Kp": 0.7, "Ki": 0.1, "Kd": 0.0},  # Left
            #    "2": {"Kp": 0.55, "Ki": 0.1, "Kd": 0.0},  # Right
            #    "3": {"Kp": 1.0, "Ki": 0.10, "Kd": 0.0},  # Straight
            #    "4": {"Kp": 1.0, "Ki": 0.50, "Kd": 0.0},  # Follow
            #}


        self.steer_points = steer_points
        self.turn_control = CustomController(pid)
        self.speed_control = PIDController(K_P=.8, K_I=.08, K_D=0.)

        self.engine_brake_threshold = 2
        self.brake_threshold = 2

        self.last_brake = -1
        self.use_cv = use_cv
        self.normalize_rgb = common.NormalizeV2(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def run_step(self, observations, teaching=False):
        rgb = observations['rgb'].copy()
        speed = np.linalg.norm(observations['velocity'])
        _cmd = int(observations['command'])
        command = self.one_hot[int(observations['command']) - 1]

        with torch.no_grad():
            _rgb = self.transform(rgb).to(self.device).unsqueeze(0)
            _speed = torch.FloatTensor([speed]).to(self.device)
            _command = command.to(self.device).unsqueeze(0)

            if self.use_cv:
                semseg = self.transform(get_segmentation_tensor(observations['semseg'].copy(),
                                                                classes=DEFAULT_CLASSES)).float().to(self.device)
                depth = self.transform(observations['depth'].copy() / 255).float().to(self.device)
                _rgb = self.normalize_rgb(_rgb)
                # Stack RGB, semseg and depth.
                # Need to unsqueeze in order to make prediction, because 'image' must be a batch with a single instance
                image = torch.cat([_rgb.squeeze(), semseg, depth], dim=0).unsqueeze(0)
            else:
                image = _rgb

            if self.model.all_branch:
                model_pred, _ = self.model(image, _speed, _command)
            else:
                model_pred = self.model(image, _speed, _command)

        model_pred = model_pred.squeeze().detach().cpu().numpy()
        
        pixel_pred = model_pred

        # Project back to world coordinate
        model_pred = (model_pred+1)*self.img_size/2

        world_pred = self.unproject(model_pred)

        targets = [(0, 0)]

        for i in range(STEPS):
            pixel_dx, pixel_dy = world_pred[i]
            angle = np.arctan2(pixel_dx, pixel_dy)
            dist = np.linalg.norm([pixel_dx, pixel_dy])

            targets.append([dist * np.cos(angle), dist * np.sin(angle)])

        targets = np.array(targets)

        target_speed = np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() / (self.gap * DT)
        
        c, r = ls_circle(targets)
        n = self.steer_points.get(str(_cmd), 1)
        closest = common.project_point_to_circle(targets[n], c, r)
        
        acceleration = target_speed - speed

        v = [1.0, 0.0, 0.0]
        w = [closest[0], closest[1], 0.0]
        alpha = common.signed_angle(v, w)

        steer = self.turn_control.run_step(alpha, _cmd)
        throttle = self.speed_control.step(acceleration)
        brake = 0.0

        # Slow or stop.
        
        if target_speed <= self.engine_brake_threshold:
            steer = 0.0
            throttle = 0.0
        
        if target_speed <= self.brake_threshold:
            brake = 1.0
            
        self.debug = {
                # 'curve': curve,
                'target_speed': target_speed,
                'target': closest,
                'locations_world': targets,
                'locations_pixel': model_pred.astype(int),
                }
        
        control = self.postprocess(steer, throttle, brake)
        if teaching:
            return control, pixel_pred
        else:
            return control

    def unproject(self, output, world_y=1.4, fov=90):

        cx, cy = self.img_size / 2
        
        w, h = self.img_size
        
        f = w /(2 * np.tan(fov * np.pi / 360))
        
        xt = (output[...,0:1] - cx) / f
        yt = (output[...,1:2] - cy) / f
        
        world_z = world_y / yt
        world_x = world_z * xt
        
        world_output = np.stack([world_x, world_z],axis=-1)
        
        if self.fixed_offset:
            world_output[...,1] -= self.fixed_offset
        
        world_output = world_output.squeeze()
        
        return world_output
