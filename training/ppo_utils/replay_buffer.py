import numpy as np
import torch
from torchvision import transforms


class PPOReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, buffer_limit=100000):
        self.buffer_limit = buffer_limit
        self._data = []
        self._rewards = []
        self._is_terminals = []
        self._state_values = []

        self.rgb_transform = transforms.ToTensor()

        self.normalized = False

    def __len__(self):
        return len(self._data)

    def __getitem__(self, _idx):
        state, action, action_logprobs, reward, is_terminal = self._data[_idx]
        state["rgb_img"] = self.rgb_transform(state["rgb_img"])
        return _idx, state, action, action_logprobs, reward, is_terminal

    def clear_buffer(self):
        self._data = []
        self._rewards = []
        self._is_terminals = []
        self._state_values = []

    def add_data(self, state, action, action_logprobs, reward, is_terminal):
        self.normalized = False
        self._data.append((state, action, action_logprobs, reward, is_terminal))
        self._rewards.append(reward)
        self._is_terminals.append(is_terminal)
        self._state_values.append(state["state_value"])

        if len(self._data) > self.buffer_limit:
            # Pop the one with lowest loss
            idx = np.argsort(self._weights)[0]
            self._data.pop(idx)
            self._weights.pop(idx)

    def get_rewards(self):
        return self._rewards

    def get_is_terminals(self):
        return self._is_terminals

    def get_state_values(self):
        return self._state_values
