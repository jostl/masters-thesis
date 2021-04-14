import glob
import os
import sys

try:
    sys.path.append(glob.glob('PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("could not find the CARLA egg")
    pass
try:
    sys.path.append(glob.glob('../PythonAPI')[0])
    sys.path.append(glob.glob('../bird_view')[0])
    sys.path.append(glob.glob('../')[0])
except IndexError as e:
    pass
import argparse
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import random

from benchmark import make_suite
from bird_view.utils import carla_utils as cu
from training.phase2_utils import setup_image_model
from training.ppo_utils.agent import PPOImageAgent
from training.ppo_utils.critic import BirdViewCritic
from training.ppo_utils.replay_buffer import PPOReplayBuffer

ACTOR_BACKBONE = 'resnet34'
CRITIC_BACKBONE = 'resnet18'
GAP = 5
N_STEP = 5
CROP_SIZE = 192
MAP_SIZE = 320
SAVE_EPISODES = list(range(20))


def crop_birdview(birdview, dx=0, dy=0):
    x = 260 - CROP_SIZE // 2 + dx
    y = MAP_SIZE // 2 + dy

    birdview = birdview[
               x - CROP_SIZE // 2:x + CROP_SIZE // 2,
               y - CROP_SIZE // 2:y + CROP_SIZE // 2]

    return birdview


def one_hot(x, num_digits=4, start=1):
    N = x.size()[0]
    x = x.long()[:, None] - start
    x = torch.clamp(x, 0, num_digits - 1)
    y = torch.FloatTensor(N, num_digits)
    y.zero_()
    y.scatter_(1, x, 1)
    return y


def rollout(replay_buffer, image_agent, critic, max_rollout_length=4000, port=2000, planner="new", **kwargs):
    progress = tqdm(range(max_rollout_length), desc='Frame')
    weather = np.random.choice(list(cu.TRAIN_WEATHERS.keys()))
    data = []
    with make_suite('NoCrashTown01-v1', port=port, planner=planner) as env:
        start, target = env.pose_tasks[np.random.randint(len(env.pose_tasks))]
        env_params = {
            'weather': weather,
            'start': start,
            'target': target,
            'n_pedestrians': random.randint(100, 250),
            'n_vehicles': random.randint(60, 120),
        }

        env.init(**env_params)
        env.success_dist = 5.0
        env.tick()
        i = 0
        total_rewards = 0
        while not env.is_success() and not env.collided and not env.traffic_tracker.ran_light and \
                len(data) <= max_rollout_length:

            state = env.get_observations()
            control, action, action_logprobs = image_agent.run_step(state)

            env.apply_control(control)
            env.tick()

            rgb_img = state["rgb"].copy()
            command = int(state["command"])
            speed = np.linalg.norm(state["velocity"])

            birdview_img = crop_birdview(state["birdview"], dx=-10)
            state_value = critic.evaluate(*critic.prepare_data(birdview_img, speed, command))

            reward = env.get_reward(speed, **kwargs)
            is_terminal = env.collided or env.is_failure() or env.is_success() or env.traffic_tracker.ran_light

            if i % 30 == 0:
                env.move_spectator_to_player()

            data.append({
                'state': {
                    'rgb_img': rgb_img,
                    'command': command,
                    'speed': speed,
                    'birdview_img': birdview_img,
                    'state_value': state_value
                },
                'action': action,
                'action_logprobs': action_logprobs,
                'reward': reward,
                'is_terminal': is_terminal
            })
            total_rewards += reward
            progress.update(1)
            i += 1

            if len(data) >= max_rollout_length:
                break

        print("Collided: ", env.collided)
        print("Success: ", env.is_success())

        env_settings = env._world.get_settings()
        env_settings.no_rendering_mode = True
        env._world.apply_settings(env_settings)
    for datum in data:
        replay_buffer.add_data(**datum)
    return total_rewards


def update(log_dir, replay_buffer, image_agent, optimizer, device, episode, critic, critic_criterion,
           epoch_per_episode=5,
           gamma=0.99, lmbda=0.5,
           clip_ratio=0.05, batch_size=24):
    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32).to(device)

    # Retrieve rewards, terminal states and state values from the rollout(s)
    rewards = replay_buffer.get_rewards()
    terminals = replay_buffer.get_is_terminals()
    values = to_tensor(replay_buffer.get_state_values())

    # Calculate advantages using generalized advantage estimation (GAE)
    advantages = to_tensor(gae(rewards, terminals, values, gamma=gamma, lmbda=lmbda, normalize=True))

    # Calculate rewards-to-go, used later when calculating loss for the critic
    rewards_to_go = to_tensor(rtgs(rewards, terminals, normalize=True))

    loader = torch.utils.data.DataLoader(replay_buffer, batch_size=batch_size, num_workers=0, pin_memory=False,
                                         shuffle=True, drop_last=True)
    for epoch in range(epoch_per_episode):
        for i, (idxes, old_states, old_actions, old_logprobs, _, _) in tqdm(enumerate(loader)):
            # Unpack old_states into RGB, birdview, speed and command.
            rgb = old_states["rgb_img"].to(device).float()
            birdview_img = old_states["birdview_img"].to(device).float()
            speed = old_states["speed"].to(device).float()
            command = one_hot(old_states["command"]).to(device).float()

            # Calculate log probability of old actions on the new policy and retrieve entropy of distribution
            logprobs, dist_entropy = image_agent.evaluate(rgb, speed, command, old_actions)

            # Calculate ratio between new and old policy
            ratios = torch.exp(logprobs - old_logprobs)

            # Evaluate state values with the critic
            state_values = critic(birdview_img, speed, command).squeeze()

            # PPO objective function
            surr1 = ratios * advantages[idxes]
            surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages[idxes]
            loss = -torch.min(surr1, surr2) + \
                   0.5 * critic_criterion(state_values, rewards_to_go[idxes]) - 0.01 * dist_entropy

            # Take gradient step
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

    # Set the new policy as the old policy
    image_agent.policy_old.load_state_dict(image_agent.model.state_dict())

    if episode in SAVE_EPISODES:
        torch.save(image_agent.model.state_dict(),
                   str(Path(config['log_dir']) / ('model-%d.th' % episode)))


def gae(rewards, terminals, values, gamma, lmbda, normalize=False):
    # Calculates generalized advantage estimation (GAE).
    # Code based on https://nn.labml.ai/rl/ppo/gae.html
    n_advantages = len(rewards)
    advantages = np.zeros(n_advantages)
    last_advantage = 0
    last_value = values[-1]
    for t in reversed(range(n_advantages)):
        mask = 1 - terminals[t]
        last_value = last_value * mask
        last_advantage = last_advantage * mask
        delta = rewards[t] + gamma * last_value - values[t]
        last_advantage = delta + gamma * lmbda * last_advantage
        advantages[t] = last_advantage
        last_value = values[t]
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
    return advantages


def rtgs(rewards, terminals, normalize=False):
    # Calculates rewards-to-go
    # Code based on https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/2_rtg_pg.py
    n = len(rewards)
    rewards_to_go = np.zeros_like(rewards)
    for i in reversed(range(n)):
        # Logic: 'i' is an index for a state.
        # Only add rewards-to-go if 'i + 1' exist and 'i' is not a terminal state
        rewards_to_go[i] = rewards[i] + (rewards_to_go[i + 1] if i + 1 < n and not terminals[i] else 0)
    if normalize:
        rewards_to_go = (rewards_to_go - np.mean(rewards_to_go)) / (np.std(rewards_to_go) + 1e-7)
    return rewards_to_go


def main():
    config = ConfigParser()
    config.read("training/ppo_config/template.cfg")

    # SETUP
    log_dir = str(config["SETUP"]["log_dir"])
    port = int(config["SETUP"]["port"])
    device = str(config["SETUP"]["device"])
    batch_size = int(config["SETUP"]["batch_size"])
    num_workers = int(config["SETUP"]["num_workers"])
    use_cv = str(config["SETUP"]["use_cv"]) == "True"
    resume = str(config["SETUP"]["resume"]) == "True"

    # TRAINING (Hyperparameters)
    max_episode = int(config["TRAINING"]["max_episode"])
    max_rollout_length = int(config["TRAINING"]["max_rollout_length"])
    rollouts_per_episode = int(config["TRAINING"]["rollouts_per_episode"])
    epoch_per_episode = int(config["TRAINING"]["epoch_per_episode"])
    clip_ratio = float(config["TRAINING"]["clip_ratio"])
    gamma = float(config["TRAINING"]["gamma"])
    lmbda = float(config["TRAINING"]["lambda"])

    # REWARD
    alpha = float(config["REWARD"]["alpha"])
    beta = float(config["REWARD"]["beta"])
    phi = float(config["REWARD"]["phi"])
    delta = float(config["REWARD"]["delta"])

    # AGENT
    action_std = float(config["AGENT"]["action_std"])
    min_action_std = float(config["AGENT"]["min_action_std"])
    action_std_decay_rate = float(config["AGENT"]["decay_rate"])
    action_std_decay_frequency = float(config["AGENT"]["decay_frequency"])

    # ACTOR
    actor_ckpt = str(config["ACTOR"]["actor_ckpt"])
    actor_lr = float(config["ACTOR"]["learning_rate"])
    actor_imagenet_pretrained = str(config["ACTOR"]["imagenet_pretrained"]) == True
    actor_backbone = str(config["ACTOR"]["backbone"])
    use_trained_cv = str(config["ACTOR"]) == "True"

    # CRITIC
    critic_ckpt = str(config["CRITIC"]["critic_ckpt"])
    critic_lr = float(config["CRITIC"]["learning_rate"])
    critic_backbone = str(config["CRITIC"]["backbone"])

    # Check if everything is legal
    assert not resume and actor_ckpt, "Resuming training requires actor checkpoint. " \
                                      "Actor path is empty."

    assert not resume and critic_ckpt, "Resuming training requires critic checkpoint. " \
                                       "Critic path is empty."

    assert use_cv != use_trained_cv, 'Cannot use ground truth and "trained" computer vision at the same time.'

    replay_buffer = PPOReplayBuffer()
    reward_params = {'alpha': alpha, 'beta': beta, 'phi': phi, 'delta': delta}

    # Setup actor networks
    actor_net = setup_image_model(backbone=actor_backbone, image_ckpt=actor_ckpt, device=device,
                                  imagenet_pretrained=actor_imagenet_pretrained, all_branch=True)
    actor_net_old = setup_image_model(backbone=actor_backbone, image_ckpt=actor_ckpt, device=device,
                                      imagenet_pretrained=actor_imagenet_pretrained)
    # Setup agent
    image_agent_kwargs = {
        'camera_args': {'w': 384, 'h': 160, 'fov': 90, 'world_y': 1.4, 'fixed_offset': 4.0}}

    image_agent = PPOImageAgent(model=actor_net, policy_old=actor_net_old, action_std=action_std,
                                min_action_std=min_action_std, action_std_decay_rate=action_std_decay_rate,
                                **image_agent_kwargs)

    # Setup critic network and criterion
    critic_net = BirdViewCritic(backbone=critic_backbone, device=device)
    if critic_ckpt:
        critic_net.load_state_dict(torch.load(critic_ckpt))
    critic_criterion = nn.MSELoss()

    # Setup optimizers
    optimizer = torch.optim.Adam([{'params': actor_net.parameters(), 'lr': actor_lr},
                                  {'params': critic_net.parameters(), 'lr': critic_lr}])

    """
     ======================
         MAIN PPO LOOP 
     ======================
    """
    for episode in range(max_episode):
        for i in range(rollouts_per_episode):
            print("Episode ", episode + 1, ", Rollout ", i + 1)
            rewards = rollout(replay_buffer, image_agent, critic_net, max_rollout_length,
                              port=port, **reward_params)
        update(log_dir, replay_buffer, image_agent, optimizer, device, episode, critic_net, critic_criterion,
               epoch_per_episode, gamma=gamma, lmbda=lmbda, clip_ratio=clip_ratio, batch_size=batch_size)

        image_agent.decay_action_std()
        replay_buffer.clear_buffer()


if __name__ == '__main__':
    main()
