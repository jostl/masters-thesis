import glob
import sys

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from bird_view.models import common
from bird_view.models.agent import Agent
from bird_view.models.controller import CustomController, PIDController
from bird_view.models.controller import ls_circle
from utils.train_utils import one_hot

try:
    sys.path.append(glob.glob('../../PythonAPI')[0])
    sys.path.append(glob.glob('../../bird_view')[0])
except IndexError as e:
    pass

CROP_SIZE = 192
PIXELS_PER_METER = 5

STEPS = 5
COMMANDS = 4
DT = 0.1


class PPOImageAgent(Agent):
    def __init__(self, policy_old, steer_points=None, pid=None, gap=5,
                 camera_args={'x': 384, 'h': 160, 'fov': 90, 'world_y': 1.4, 'fixed_offset': 4.0}, action_std=0.01,
                 min_action_std=0.001, action_std_decay_rate = 0.0001, action_std_decay_frequency=10000,
                 **kwargs):
        super().__init__(**kwargs)

        # 'self.model' is the current policy being updated
        self.policy_old = policy_old

        self.fixed_offset = float(camera_args['fixed_offset'])
        print("Offset: ", self.fixed_offset)
        w = float(camera_args['w'])
        h = float(camera_args['h'])
        self.img_size = np.array([w, h])
        self.gap = gap

        if steer_points is None:
            # original LBC steer points
            steer_points = {"1": 4, "2": 3, "3": 2, "4": 2}

            # reproduction wrong_pid steer points
            #steer_points = {"1": 4, "2": 3, "3": 2, "4": 3}

        if pid is None:
            # original LBC pid
            pid = {
                "1": {"Kp": 0.5, "Ki": 0.20, "Kd": 0.0},  # Left
                "2": {"Kp": 0.7, "Ki": 0.10, "Kd": 0.0},  # Right
                "3": {"Kp": 1.0, "Ki": 0.10, "Kd": 0.0},  # Straight
                "4": {"Kp": 1.0, "Ki": 0.50, "Kd": 0.0},  # Follow
            }

            # Reproduction_wrong_pid model-10 pid
            #pid = {
            #    "1": {"Kp": 0.7, "Ki": 0.25, "Kd": 0.0},  # Left
            #    "2": {"Kp": 0.7, "Ki": 0.10, "Kd": 0.0},  # Right
            #    "3": {"Kp": 1.0, "Ki": 0.10, "Kd": 0.0},  # Straight
            #    "4": {"Kp": 0.75, "Ki": 0.45, "Kd": 0.0},  # Follow
            #}

        self.steer_points = steer_points
        self.turn_control = CustomController(pid)
        self.speed_control = PIDController(K_P=.8, K_I=.08, K_D=0.)

        # original LBC thresholds
        self.engine_brake_threshold = 1.2
        self.brake_threshold = 1.2

        # reproduction wrong_pid thresholds
        #self.engine_brake_threshold = 1.5
        #self.brake_threshold = 1.5

        self.last_brake = -1

        a = np.linspace(min_action_std, action_std, 5)
        b = np.zeros((5, 2))
        b[:, 0] = a
        b[:, 1] = a
        b = np.ndarray.flatten(b)
        b = b * b
        self.action_var = torch.from_numpy(b).float()
        self.action_var2 = torch.full((10,), action_std * action_std)
        self.action_std = action_std
        self.min_action_std = min_action_std
        self.action_std_decay_rate = action_std_decay_rate
        self.action_std_decay_frequency = action_std_decay_frequency

    def run_step(self, observations):
        rgb = observations['rgb'].copy()
        speed = np.linalg.norm(observations['velocity'])
        _cmd = int(observations['command'])
        command = self.one_hot[int(observations['command']) - 1]

        with torch.no_grad():
            _rgb = self.transform(rgb).to(self.device).unsqueeze(0)
            _speed = torch.FloatTensor([speed]).to(self.device)
            _command = command.to(self.device).unsqueeze(0)
            if self.policy_old.all_branch:
                model_pred, _ = self.policy_old(_rgb, _speed, _command)
            else:
                model_pred = self.policy_old(_rgb, _speed, _command)

        action, action_logprob = self.sample_action(model_pred)
        model_pred = action.squeeze().detach().cpu().numpy()

        # Project back to world coordinate
        model_pred = (model_pred + 1) * self.img_size / 2

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
        return control, action, action_logprob

    def unproject(self, output, world_y=1.4, fov=90):

        cx, cy = self.img_size / 2

        w, h = self.img_size

        f = w / (2 * np.tan(fov * np.pi / 360))

        xt = (output[..., 0:1] - cx) / f
        yt = (output[..., 1:2] - cy) / f

        world_z = world_y / yt
        world_x = world_z * xt

        world_output = np.stack([world_x, world_z], axis=-1)

        if self.fixed_offset:
            world_output[..., 1] -= self.fixed_offset

        world_output = world_output.squeeze()

        return world_output

    def evaluate(self, rgb, speed, command, action):
        # Calculate mean actions
        if self.model.all_branch:
            action_mean, _ = self.model(rgb, speed, command)
        else:
            action_mean = self.model(rgb, speed, command)
        # Shape: (batch_size, n_waypoints, 2)
        batch_size, n_waypoints, _ = action_mean.shape

        # Reshape action_mean to shape (batch_size, 10)
        action_mean_view = action_mean.squeeze().view((batch_size, n_waypoints * 2))

        # Make a 2-D tensor of shape (batch_size, 10), where each vector is 'self.action_var'
        action_var = self.action_var.expand_as(action_mean_view)

        # Makes a 3-D tensor with shape (batch_size, 10, 10)
        # Each 10 x 10 matrix is a covariance matrix with entries only along the diagonal
        cov_mat = torch.diag_embed(action_var).to(self.device)

        # Create a multivariate normal distribution
        dist = MultivariateNormal(action_mean_view, cov_mat)

        # Get the log of the probability density function evaluated at each 'action' in this batch
        action_logprob = dist.log_prob(action.squeeze().view((batch_size, n_waypoints * 2)))

        _action_mean_view = action_mean_view.cpu().detach().numpy()[0]
        _action_logprob = action_logprob.cpu().detach().numpy()[0]

        # Get the entropy of the distribution
        dist_entropy = dist.entropy()

        return action_logprob, dist_entropy

    def sample_action(self, model_pred):
        action_mean = model_pred  # mu (mean value)
        original_shape = action_mean.shape

        # Reshape action_mean to (1, 10)
        action_mean_view = action_mean.squeeze().view((10,))

        # Make a covariance matrix filled the vector 'self.action_var' along the diagonal
        cov_mat = torch.diag(self.action_var).to(self.device)
        # Make a multivariate normal-dist with 'action_mean' as vector of means, and 'cov_mat' as covariance matrix
        dist = MultivariateNormal(action_mean_view, cov_mat)

        # Sample an action
        action = dist.sample()

        # Get the log of the probability density function evaluated at 'action'
        action_logprob = dist.log_prob(action)

        # Reshape the action into original shape
        action = action.view(original_shape)
        return action, action_logprob

    def decay_action_std(self):
        self.action_std = max(self.action_std - self.action_std_decay_rate, self.min_action_std)
        self.action_std = round(self.action_std, 4)
        a = np.linspace(self.min_action_std, self.action_std, 5)
        b = np.zeros((5, 2))
        b[:, 0] = a
        b[:, 1] = a
        b = np.ndarray.flatten(b)
        b = b * b
        self.action_var = torch.from_numpy(b).float()
