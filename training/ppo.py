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
from configparser import ConfigParser
from pathlib import Path

import carla

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
from training.ppo_utils.helpers import rtgs, gae, _paint, get_reward
from tensorboardX import SummaryWriter

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


def rollout(replay_buffer, image_agent, critic, episode, max_rollout_length=4000, rollouts_per_episode=5, port=2000,
            planner="new", show=False, include_hero=False, **kwargs):
    progress = tqdm(range(max_rollout_length * rollouts_per_episode), desc='Frame')
    data = [[] for _ in range(rollouts_per_episode)]
    with make_suite('NoCrashTown01-v1', port=port, planner=planner) as env:
        for i in range(rollouts_per_episode):
            print("Episode", episode, ", rollout", i)
            start, target = env.pose_tasks[np.random.randint(len(env.pose_tasks))]
            env_params = {
                'weather': np.random.choice(list(cu.TRAIN_WEATHERS.keys())),
                'start': start,
                'target': target,
                'n_pedestrians': random.randint(100, 250),
                'n_vehicles': random.randint(60, 120),
            }

            env.init(**env_params)
            env.success_dist = 5.0
            env.tick()
            time_steps = 0
            total_rewards = 0
            while not env.is_success() and not env.collided and not env.traffic_tracker.ran_light and \
                    len(data[i]) <= max_rollout_length:
                observations = env.get_observations(include_hero=include_hero)

                control, action, action_logprobs = image_agent.run_step(observations)

                diagnostic = env.apply_control(control)
                env.tick()

                rgb_img = observations["rgb"].copy()
                command = int(observations["command"])
                speed = np.linalg.norm(observations["velocity"])

                birdview_img = crop_birdview(observations["birdview"], dx=-10)
                state_value = critic.evaluate(*critic.prepare_data(birdview_img, speed, command))

                reward = get_reward(env, speed, **kwargs)
                is_terminal = env.collided or env.is_failure() or env.is_success() or env.traffic_tracker.ran_light

                if time_steps % 30 == 0:
                    env.move_spectator_to_player()

                data[i].append({
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
                time_steps += 1
                if show:
                    _paint(observations, control, diagnostic, reward, image_agent.action_std, image_agent.debug, env)

                if len(data) >= max_rollout_length:
                    break

            print("Collided: ", env.collided)
            print("Success: ", env.is_success())
            env.clean_up()
        env_settings = env._world.get_settings()
        env_settings.no_rendering_mode = True
        env._world.apply_settings(env_settings)
    for rollout_data in data:
        for datum in rollout_data:
            replay_buffer.add_data(**datum)
    return total_rewards, time_steps


def update(log_dir, replay_buffer, image_agent, optimizer, device, episode, critic, critic_criterion,
           epoch_per_episode=5, gamma=0.99, lmbda=0.5, clip_ratio=0.05, c1=1.0, c2=0.01, batch_size=24, num_workers=0,
           critic_writer=None):
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

    loader = torch.utils.data.DataLoader(replay_buffer, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=False, shuffle=True, drop_last=True)

    # Connecting to server to prevent timout
    # client = carla.Client("localhost", 2000)
    # client.set_timeout(30)
    # cu.set_sync_mode(client, False)
    # world = client.load_world("town01")
    print("Training on {} examples".format(len(replay_buffer)))
    for epoch in range(epoch_per_episode):
        # world.wait_for_tick()
        desc = "Episode {}, epoch {}".format(episode, epoch)
        running_critic_loss = 0
        for i, (idxes, rgb, speed, command, birdview, old_actions, old_logprobs) in tqdm(enumerate(loader), desc=desc):
            # Unpack old_states into RGB, birdview, speed and command.
            rgb = rgb.to(device).float()
            birdview = birdview.to(device).float()
            speed = speed.to(device).float()
            command = one_hot(command).to(device).float()

            # Calculate log probability of old actions on the new policy and retrieve entropy of distribution
            logprobs, dist_entropy = image_agent.evaluate(rgb, speed, command, old_actions)

            # Calculate ratio between new and old policy
            ratios = torch.exp(logprobs - old_logprobs)

            # Evaluate state values with the critic
            state_values, _ = critic(birdview, speed, command)
            state_values = state_values.squeeze()

            # PPO objective function
            surr1 = ratios * advantages[idxes]
            surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages[idxes]
            objective = torch.min(surr1, surr2)

            # Critic loss
            critic_loss = critic_criterion(state_values, rewards_to_go[idxes])

            # Compute loss
            loss = - objective + c1 * critic_loss - c2 * dist_entropy

            # Take gradient step
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            # Running loss for critic
            running_critic_loss += critic_loss.item() * rgb.size(0)

        epoch_critic_loss = running_critic_loss / len(replay_buffer)
        if critic_writer is not None:
            if episode == 0:
                epoch_write_number = epoch
            else:
                epoch_write_number = epoch_per_episode * episode + epoch
            critic_writer.add_scalar("epoch_loss", epoch_critic_loss, epoch_write_number)
    # cu.set_sync_mode(client, False)

    # Set the new policy as the old policy
    image_agent.policy_old.load_state_dict(image_agent.model.state_dict())

    # Save models
    torch.save(image_agent.model.state_dict(), str(Path(log_dir) / ('actor-%d.th' % episode)))
    torch.save(critic.state_dict(), str(Path(log_dir) / ('critic-%d.th' % episode)))


def main():
    config = ConfigParser()
    config.read("training/ppo_config/template.cfg")

    # SETUP
    log_dir = str(config["SETUP"]["log_dir"])
    port = int(config["SETUP"]["port"])
    device = str(config["SETUP"]["device"])
    batch_size = int(config["SETUP"]["batch_size"])
    num_workers = int(config["SETUP"]["num_workers"])
    resume_episode = int(config["SETUP"]["resume_episode"])
    computer_vision = str(config["SETUP"]["computer_vision"])
    show = str(config["SETUP"]["show"]) == "True"

    # TRAINING (Hyperparameters)
    max_episode = int(config["TRAINING"]["max_episode"])
    max_rollout_length = int(config["TRAINING"]["max_rollout_length"])
    rollouts_per_episode = int(config["TRAINING"]["rollouts_per_episode"])
    epoch_per_episode = int(config["TRAINING"]["epoch_per_episode"])
    clip_ratio = float(config["TRAINING"]["clip_ratio"])
    gamma = float(config["TRAINING"]["gamma"])
    lmbda = float(config["TRAINING"]["lambda"])
    c1 = float(config["TRAINING"]["c1"])
    c2 = float(config["TRAINING"]["c2"])

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

    # CRITIC
    critic_ckpt = str(config["CRITIC"]["critic_ckpt"])
    critic_lr = float(config["CRITIC"]["learning_rate"])
    critic_backbone = str(config["CRITIC"]["backbone"])
    include_hero = str(config["CRITIC"]["include_hero"]) == "True"

    # Check if everything is legal
    assert computer_vision in {"None", "gt",
                               "trained"}, "'computer_vision' must be equal to 'None','gt'(ground truth) or 'trained'," \
                                           " found '{}'".format(computer_vision)
    assert computer_vision != "gt" and computer_vision != "trained", "Not implemented yet lol"

    if resume_episode > 0:
        assert actor_ckpt, "Resuming training requires actor checkpoint. " \
                           "Actor path is empty."

        assert critic_ckpt, "Resuming training requires critic checkpoint. " \
                            "Critic path is empty."

    # INITIALIZING
    path = Path(log_dir)
    if not path.exists():
        path.mkdir()

    replay_buffer = PPOReplayBuffer()
    reward_params = {'alpha': alpha, 'beta': beta, 'phi': phi, 'delta': delta}
    if resume_episode > 0:
        action_std = torch.load(Path(log_dir) / "action_std{}".format(resume_episode))
        resume_episode += 1
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

    # TODO: Store Agent args


    # Setup critic network and criterion
    critic_net = BirdViewCritic(backbone=critic_backbone, device=device, all_branch=True, input_channel=8).to(device)
    if critic_ckpt:
        critic_net.load_state_dict(torch.load(critic_ckpt))
    critic_criterion = nn.MSELoss()

    # Setup optimizers
    optimizer = torch.optim.Adam([{'params': actor_net.parameters(), 'lr': actor_lr},
                                  {'params': critic_net.parameters(), 'lr': critic_lr}])

    # Setup writers for tensorboard
    critic_writer = SummaryWriter(path / "logs/critic")
    reward_writer = SummaryWriter(path / "logs/reward")

    """
     ======================
         MAIN PPO LOOP 
     ======================
    """
    total_time_steps = 0
    for episode in range(resume_episode, max_episode):
        episode_rewards = 0
        rewards, time_steps = rollout(replay_buffer, image_agent, critic_net, episode, max_rollout_length,
                                      rollouts_per_episode=rollouts_per_episode,
                                      port=port, show=show, include_hero=include_hero, **reward_params)
        total_time_steps += time_steps
        episode_rewards += rewards
        reward_writer.add_scalar("episode_rewards", episode_rewards, episode)
        update(log_dir, replay_buffer, image_agent, optimizer, device, episode, critic_net, critic_criterion,
               epoch_per_episode, gamma=gamma, lmbda=lmbda, clip_ratio=clip_ratio, batch_size=batch_size,
               num_workers=num_workers, c1=c1, c2=c2, critic_writer=critic_writer)
        torch.save(image_agent.action_std, path / "action_std{}".format(episode))
        image_agent.decay_action_std()
        replay_buffer.clear_buffer()


if __name__ == '__main__':
    main()
