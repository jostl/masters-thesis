import glob
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.distributions import MultivariateNormal
from bird_view.models import common
from bird_view.models.agent import Agent
from bird_view.models.controller import CustomController, PIDController
from bird_view.models.controller import ls_circle

try:
    sys.path.append(glob.glob('../PythonAPI')[0])
    sys.path.append(glob.glob('../bird_view')[0])
except IndexError as e:
    pass

CROP_SIZE = 192
PIXELS_PER_METER = 5

STEPS = 5
COMMANDS = 4
DT = 0.1

class PPOReplayBuffer2(torch.utils.data.Dataset):
    def __init__(self, buffer_limit=100000):
        self.buffer_limit = buffer_limit
        self._data = []
        self._weights = []
        self.rgb_transform = transforms.ToTensor()

        self.normalized = False

    def __len__(self):
        return len(self._data)

    def __getitem__(self, _idx):
        state, action, action_logprobs, reward, is_terminal = self._data[_idx]

        return _idx, state, action, action_logprobs, reward, is_terminal

    def update_weights(self, idxes, losses):
        idxes = idxes.numpy()
        losses = losses.detach().cpu().numpy()
        for idx, loss in zip(idxes, losses):
            if idx > len(self._data):
                continue

            self._new_weights[idx] = loss

    def init_new_weights(self):
        self._new_weights = self._weights.copy()

    def normalize_weights(self):
        self._weights = self._new_weights
        self.normalized = True

    def add_data(self, state, action, action_logprobs, reward, is_terminal):
        self.normalized = False
        self._data.append((state, action, action_logprobs, reward, is_terminal))

        if len(self._data) > self.buffer_limit:
            # Pop the one with lowest loss
            idx = np.argsort(self._weights)[0]
            self._data.pop(idx)
            self._weights.pop(idx)

    def remove_data(self, idx):
        self._weights.pop(idx)
        self._data.pop(idx)

    def get_rewards(self):
        return self._data[:][-2]

    def get_is_terminals(self):
        return self._data[:][-1]

    def get_highest_k(self, k):
        top_idxes = np.argsort(self._weights)[-k:]
        rgb_images = []
        bird_views = []
        targets = []
        cmds = []
        speeds = []

        for idx in top_idxes:
            if idx < len(self._data):
                rgb_img, cmd, speed, target, birdview_img = self._data[idx]
                rgb_images.append(TF.to_tensor(np.ascontiguousarray(rgb_img)))
                bird_views.append(TF.to_tensor(birdview_img))
                cmds.append(cmd)
                speeds.append(speed)
                targets.append(target)

        return torch.stack(rgb_images), torch.stack(bird_views), torch.FloatTensor(cmds), torch.FloatTensor(
            speeds), torch.FloatTensor(targets)


class PPOReplayBuffer():
    def __init__(self, buffer_limit):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def add_data(self, data):
        pass

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPOImageAgent(Agent):
    def __init__(self, replay_buffer, policy, policy_old, critic, steer_points=None, pid=None, gap=5,
                 camera_args={'x': 384, 'h': 160, 'fov': 90, 'world_y': 1.4, 'fixed_offset': 4.0},
                 **kwargs):
        super().__init__(**kwargs)

        self.policy = policy
        self.policy_old = policy_old
        self.critic = critic
        self.fixed_offset = float(camera_args['fixed_offset'])
        print("Offset: ", self.fixed_offset)
        w = float(camera_args['w'])
        h = float(camera_args['h'])
        self.img_size = np.array([w, h])
        self.gap = gap

        if steer_points is None:
            steer_points = {"1": 4, "2": 3, "3": 2, "4": 2}

        if pid is None:
            pid = {
                "1": {"Kp": 0.5, "Ki": 0.20, "Kd": 0.0},  # Left
                "2": {"Kp": 0.7, "Ki": 0.10, "Kd": 0.0},  # Right
                "3": {"Kp": 1.0, "Ki": 0.10, "Kd": 0.0},  # Straight
                "4": {"Kp": 1.0, "Ki": 0.50, "Kd": 0.0},  # Follow
            }

        self.steer_points = steer_points
        self.turn_control = CustomController(pid)
        self.speed_control = PIDController(K_P=.8, K_I=.08, K_D=0.)
        self.replay_buffer = replay_buffer
        self.engine_brake_threshold = 2.0
        self.brake_threshold = 2.0

        self.last_brake = -1

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

        model_pred = model_pred.squeeze().detach().cpu().numpy()

        action = model_pred

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
        # TODO: Implementer action logprobs
        action_logprobs = None
        return control, action, action_logprobs

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

    def evaluate(self, state):
        rgb = state['rgb'].copy()
        speed = np.linalg.norm(state['velocity'])
        _cmd = int(state['command'])
        command = self.one_hot[int(state['command']) - 1]

        with torch.no_grad():
            _rgb = self.transform(rgb).to(self.device).unsqueeze(0)
            _speed = torch.FloatTensor([speed]).to(self.device)
            _command = command.to(self.device).unsqueeze(0)
            state_value = self.critic(_rgb, _speed, _command)
            if self.policy.all_branch:
                action, _ = self.policy(_rgb, _speed, _command)
            else:
                action = self.policy(_rgb, _speed, _command)

        dist_entropy = None # Todo: må få dist_entropy
        logprobs = None # todo: må få logprobs av action
        return logprobs, state_value, dist_entropy

class ActorCritic(nn.Module):
    def __init__(self, device, state_dim, action_dim, action_std):
        self.device = device
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor_net = None # TODO: Legg til ImageModell her din gjøk
        # critic
        self.critic = None # TODO: Legg til en Critic net her din idiot
        self.action_var = torch.full((action_dim,), action_std * action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy