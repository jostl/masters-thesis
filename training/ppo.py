import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
from training.ppo_utils.replay_buffer import PPOReplayBuffer
from training.phase2_utils import setup_image_model
from bird_view.utils import carla_utils as cu
from benchmark import make_suite
from tqdm import tqdm
from training.ppo_utils.agent import PPOImageAgent
from training.ppo_utils.critic import CriticNetwork

BACKBONE = 'resnet34'
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


def rollout(replay_buffer, image_agent, episode_length=4000,
            n_vehicles=100, n_pedestrians=250, port=2000, planner="new"):
    progress = tqdm(range(episode_length), desc='Frame')
    weather = np.random.choice(list(cu.TRAIN_WEATHERS.keys()))
    data = []
    while len(data) < episode_length:
        with make_suite('NoCrashTown01-v1', port=port, planner=planner) as env:
            start, target = env.pose_tasks[np.random.randint(len(env.pose_tasks))]
            env_params = {
                'weather': weather,
                'start': start,
                'target': target,
                'n_pedestrians': n_pedestrians,
                'n_vehicles': n_vehicles,
            }

            env.init(**env_params)
            env.success_dist = 5.0
            i = 0
            while not env.is_success() and not env.collided:
                env.tick()

                state = env.get_observations()
                control, action, action_logprobs = image_agent.run_step(state)

                env.apply_control(control)
                reward = env.get_reward()
                is_terminal = env.is_success() or env.collided

                if i % 50 == 0:
                    env.move_spectator_to_player()

                data.append({
                    'state': {
                        'rgb_img': state["rgb"].copy(),
                        'command': int(state["command"]),
                        'velocity': np.linalg.norm(state["velocity"])},
                    'action': action,
                    'action_logprobs': action_logprobs,
                    'reward': reward,
                    'is_terminal': is_terminal
                })

                progress.update(1)
                i += 1

                # DEBUG
                if len(data) >= episode_length:
                    break
                # DEBUG END

            print("Collided: ", env.collided)
            print("Success: ", env.is_success())

            env_settings = env._world.get_settings()
            env_settings.no_rendering_mode = True
            env._world.apply_settings(env_settings)
            if env.collided:
                data = data[:-5]
    for datum in data:
        replay_buffer.add_data(**datum)


def update(replay_buffer, image_agent, optimizer, config, episode, critic_criterion, gamma=0.99, eps_clip=0.05):
    rewards = []
    discounted_reward = 0

    for reward, is_terminal in zip(reversed(replay_buffer.get_rewards()), reversed(replay_buffer.get_is_terminals())):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        rewards.insert(0, discounted_reward)
    loader = torch.utils.data.DataLoader(replay_buffer, batch_size=10, num_workers=0, pin_memory=False,
                                         shuffle=False, drop_last=True)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(config["device"])
    for epoch in range(config['epoch_per_episode']):
        for i, (idxes, old_states, old_actions, old_logprobs, _, _) in tqdm(enumerate(loader)):
            logprobs, state_values, dist_entropy = image_agent.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards[idxes] - state_values
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * critic_criterion(state_values, rewards[idxes]) - 0.01 * dist_entropy

            # take gradient step
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

    image_agent.policy_old.load_state_dict(image_agent.model.state_dict())

    if episode in SAVE_EPISODES:
        torch.save(image_agent.model.state_dict(),
                   str(Path(config['log_dir']) / ('model-%d.th' % episode)))


def train(config):
    import utils.bz_utils as bzu

    bzu.log.init(config['log_dir'])
    bzu.log.save_config(config)

    replay_buffer = PPOReplayBuffer(**config["buffer_args"])

    actor_net = setup_image_model(**config["model_args"], device=config["device"], all_branch=True,
                                  imagenet_pretrained=False)
    actor_net_old = setup_image_model(**config["model_args"], device=config["device"], all_branch=True,
                                      imagenet_pretrained=False)


    critic_net = CriticNetwork(backbone=config["model_args"]["backbone"],
                               device=config["device"]).to(config["device"])

    image_agent_kwargs = {'camera_args': config["agent_args"]['camera_args']}
    image_agent = PPOImageAgent(replay_buffer, model=actor_net, policy_old=actor_net_old, critic=critic_net,
                                **image_agent_kwargs)

    optimizer = torch.optim.Adam(actor_net.parameters(), lr=1e-4)
    critic_criterion = nn.MSELoss()
    for episode in tqdm(range(config['max_episode']), desc='Episode'):
        rollout(replay_buffer, image_agent, episode_length=config['episode_length'],
                port=config['port'])
        # import pdb; pdb.set_trace()
        update(replay_buffer, image_agent, optimizer, config, episode, critic_criterion)
        replay_buffer.clear_buffer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--log_iterations', default=100)
    parser.add_argument('--max_episode', type=int, default=20)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--epoch_per_episode', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--speed_noise', type=float, default=0.0)

    parser.add_argument('--actor-ckpt', required=True)
    parser.add_argument('--perception-ckpt', default="")
    parser.add_argument('--n_semantic_classes', type=int, default=6)

    parser.add_argument('--fixed_offset', type=float, default=4.0)

    # Optimizer.
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-4)

    # Misc
    parser.add_argument('--port', type=int, default=2000)

    parsed = parser.parse_args()

    config = {
        'port': parsed.port,
        'log_dir': parsed.log_dir,
        'log_iterations': parsed.log_iterations,
        'batch_size': parsed.batch_size,
        'max_episode': parsed.max_episode,
        'episode_length': parsed.episode_length,
        'speed_noise': parsed.speed_noise,
        'epoch_per_episode': parsed.epoch_per_episode,
        'device': 'cuda',
        'optimizer_args': {'lr': parsed.actor_lr},
        'buffer_args': {
            'buffer_limit': 200000,
        },
        'model_args': {
            'model': 'image_ss',
            'backbone': BACKBONE,
            'image_ckpt': parsed.actor_ckpt,
            'perception_ckpt': parsed.perception_ckpt,
            'n_semantic_classes': parsed.n_semantic_classes
        },
        'agent_args': {
            'camera_args': {
                'w': 384,
                'h': 160,
                'fov': 90,
                'world_y': 1.4,
                'fixed_offset': parsed.fixed_offset,
            }
        }
    }

    train(config)
