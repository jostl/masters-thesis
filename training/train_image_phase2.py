import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm

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

from bird_view.utils import carla_utils as cu
from utils.train_utils import one_hot
from benchmark import make_suite
from perception.utils.segmentation_labels import DEFAULT_CLASSES
from perception.utils.helpers import get_segmentation_tensor
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
            x-CROP_SIZE//2:x+CROP_SIZE//2,
            y-CROP_SIZE//2:y+CROP_SIZE//2]

    return birdview


def get_control(agent_control, teacher_control, episode, beta=0.95):
    """
    A learning schedule to choose between agent control and teacher control
    lim_{episode->inf} P(control=agent_control|episode) = 1
    to make sure it converges
    """
    prob = 0.5 + 0.5*(1-beta**episode)

    if np.random.uniform(0,1) < prob:
        control = agent_control
    else:
        control = teacher_control

    return control


def rollout(replay_buffer, coord_converter, net, teacher_net, episode,
        image_agent_kwargs=dict(), birdview_agent_kwargs=dict(),
        episode_length=1000,
        n_vehicles=100, n_pedestrians=250, port=2000, planner="new", use_cv=False):

    from models.image import ImageAgent
    from models.birdview import BirdViewAgent

    decay = np.array([0.7**i for i in range(5)])
    xy_bias = np.array([0.7,0.3])

    weathers = list(cu.TRAIN_WEATHERS.keys())

    def _get_weight(a, b):
        loss_weight = np.mean((np.abs(a - b)*xy_bias).sum(axis=-1)*decay, axis=-1)
        x_weight = np.maximum(
            np.mean(a[...,0],axis=-1),
            np.mean(a[...,0]*-1.4,axis=-1),
        )

        return loss_weight

    num_data = 0
    progress = tqdm.tqdm(range(episode_length*len(weathers)), desc='Frame')
    for weather in weathers:

        data = list()

        while len(data) < episode_length:

            with make_suite('NoCrashTown01-v1', port=port, planner=planner, use_cv=use_cv) as env:

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

                image_agent_kwargs['model'] = net
                image_agent_kwargs['use_cv'] = use_cv
                birdview_agent_kwargs['model'] = teacher_net

                image_agent = ImageAgent(**image_agent_kwargs)
                birdview_agent = BirdViewAgent(**birdview_agent_kwargs)
                i = 0
                while not env.is_success() and not env.collided:
                    env.tick()
                    
                    i += 1
                    if i % 50 == 0:
                        env.move_spectator_to_player()

                    observations = env.get_observations()

                    data_dict = {}
                    if use_cv:
                        data_dict.update({
                            'semseg': observations['semseg'].copy(),
                            'depth': observations['depth'].copy()
                        })

                    image_control, _image_points = image_agent.run_step(observations, teaching=True)
                    _image_points = coord_converter(_image_points)
                    image_points = _image_points/(0.5*CROP_SIZE)-1
                    birdview_control, birdview_points = birdview_agent.run_step(observations, teaching=True)
                    weight = _get_weight(birdview_points, image_points)

                    control = get_control(image_control, birdview_control, episode)

                    env.apply_control(control)
                    data_dict.update({
                        'rgb_img': observations["rgb"].copy(),
                        'cmd': int(observations["command"]),
                        'speed': np.linalg.norm(observations["velocity"]),
                        'target': birdview_points,
                        'weight': weight,
                        'birdview_img': crop_birdview(observations['birdview'], dx=-10),
                    })
                    data.append(data_dict)

                    progress.update(1)

                    # DEBUG
                    if len(data) >= episode_length:
                        break
                    # DEBUG END

                print ("Collided: ", env.collided)
                print ("Success: ", env.is_success())

                env_settings = env._world.get_settings()
                env_settings.no_rendering_mode = True
                env._world.apply_settings(env_settings)
                if env.collided:
                    data = data[:-5]

        for datum in data:
            replay_buffer.add_data(**datum)
            num_data += 1


def _train(replay_buffer, net, teacher_net, criterion, coord_converter, logger, config, episode):

    from training.phase2_utils import _log_visuals, get_weight, repeat

    import torch.distributions as tdist
    #noiser = tdist.Normal(torch.tensor(0.0), torch.tensor(config['speed_noise']))

    teacher_net.eval()

    for epoch in range(config['epoch_per_episode']):

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

        net.train()
        replay_buffer.init_new_weights()
        loader = torch.utils.data.DataLoader(replay_buffer, batch_size=config['batch_size'], num_workers=2, pin_memory=True, shuffle=True, drop_last=True)

        description = "Epoch " + str(epoch)
        for i, (idxes, image, command, speed, target, birdview) in tqdm.tqdm(enumerate(loader), desc=description):
            #if i % 100 == 0:
                #print ("ITER: %d"%i)
            image = image.to(config['device']).float()

            birdview = birdview.to(config['device']).float()
            command = one_hot(command).to(config['device']).float()
            speed = speed.to(config['device']).float()

            if config['speed_noise'] > 0:
                speed += noiser.sample(speed.size()).to(speed.device)
                speed = torch.clamp(speed, 0, 10)

            if len(image.size()) > 4:
                B, batch_aug, c, h, w = image.size()
                image = image.view(B*batch_aug,c,h,w)
                birdview = repeat(birdview, batch_aug)
                command = repeat(command, batch_aug)
                speed = repeat(speed, batch_aug)
            else:
                B = image.size(0)
                batch_aug = 1

            with torch.no_grad():
                _teac_location, _teac_locations = teacher_net(birdview, speed, command)
            if config["trained_cv"]:
                (_pred_location, _pred_locations), _ = net(image, speed, command)
            else:
                _pred_location, _pred_locations = net(image, speed, command)
            pred_location = coord_converter(_pred_location)
            pred_locations = coord_converter(_pred_locations)

            optimizer.zero_grad()
            loss = criterion(pred_locations, _teac_locations)

            # Compute resample weights
            pred_location_normed = pred_location / (0.5*CROP_SIZE) - 1.
            weights = get_weight(pred_location_normed, _teac_location)
            weights = torch.mean(torch.stack(torch.chunk(weights,B)),dim=0)


            replay_buffer.update_weights(idxes, weights)

            loss_mean = loss.mean()

            loss_mean.backward()
            optimizer.step()

            should_log = False
            should_log |= i % config['log_iterations'] == 0

            if should_log:
                metrics = dict()
                metrics['loss'] = loss_mean.item()
                if image.shape[1] > 3:
                    rgb_image = image[:, 0:3]
                else:
                    rgb_image = image
                images = _log_visuals(
                    rgb_image, birdview, speed, command, loss,
                    pred_location, (_pred_location+1)*coord_converter._img_size/2, _teac_location)

                logger.scalar(loss_mean=loss_mean.item())
                logger.image(birdview=images)

        replay_buffer.normalize_weights()

        image, birdview, command, speed, target = replay_buffer.get_highest_k(32)

        image = image.to(config['device']).float()
        birdview = birdview.to(config['device']).float()
        command = one_hot(command).to(config['device']).float()
        speed = speed.to(config['device']).float()

        with torch.no_grad():
            _teac_location, _teac_locations = teacher_net(birdview, speed, command)

        net.eval()
        _pred_location, _pred_locations = net(image, speed, command)
        pred_location = coord_converter(_pred_location)
        pred_locations = coord_converter(_pred_locations)
        pred_location_normed = pred_location / (0.5*CROP_SIZE) - 1.
        weights = get_weight(pred_location_normed, _teac_location)

        # TODO: Plot highest
        if image.shape[1] > 3:
            rgb_image = image[:, 0:3]
        else:
            rgb_image = image
        images = _log_visuals(
            rgb_image, birdview, speed, command, weights,
            pred_location, (_pred_location+1)*coord_converter._img_size/2, _teac_location)

        logger.image(topk=images)

        logger.end_epoch()

    if episode in SAVE_EPISODES:
        torch.save(net.state_dict(),
            str(Path(config['log_dir']) / ('model-%d.th' % episode)))


def train(config):
    use_cv = config["agent_args"]["use_cv"]
    trained_cv = config["agent_args"]["trained_cv"]
    assert not (use_cv and trained_cv), \
        "Cannout use ground truth CV and trained CV at the same time."
    assert not(use_cv and (config["buffer_args"]["batch_aug"] > 1)), \
        "Currently not legal to have batch aug > 1 and use_cv = True."
    import utils.bz_utils as bzu
    from training.phase2_utils import (
        CoordConverter,
        ReplayBuffer,
        ReplayBufferDisk,
        LocationLoss,
        load_birdview_model,
        setup_image_model
    )
    buffer_type = config["buffer_args"]["type"]
    if buffer_type == "standard":
        replay_buffer = ReplayBuffer(**config["buffer_args"], use_cv=use_cv)
    elif buffer_type == "disk" and config["resume_episode"] < 0:
        replay_buffer = ReplayBufferDisk(path=config["log_dir"], use_cv=use_cv, **config["buffer_args"])

    if config['resume_episode'] >= 0:
        print("Resuming from previous run. Setting up replay buffer from episode {}".format(config["resume_episode"]))
        if buffer_type == "standard":
            print("Loading images, cmds, speeds and targets")
            image_data = torch.load(
            Path(config['log_dir']) / 'replay_buffer-{}_image_data.saved'.format(config["resume_episode"]))
            print("Loading birdview images")
            birdview_data = torch.load(
                Path(config['log_dir']) / 'replay_buffer-{}_birdview_data.saved'.format(config["resume_episode"]))

            assert len(birdview_data) == len(image_data), "Length of image-data and birdview-data is not the same."

            print("Loading replay buffer weights")
            replay_buffer_weights = torch.load(
                Path(config['log_dir']) / 'replay_buffer-{}_weights.saved'.format(config["resume_episode"]))

            assert len(replay_buffer_weights) == len(image_data), "dint find enough weights"

            for i in range(len(image_data)):
                input_data, cmd, speed, target = image_data[i]
                birdview = birdview_data[i]
                weight = replay_buffer_weights[i]
                if use_cv:
                    rgb, semseg, depth = input_data
                    replay_buffer.add_data(rgb, cmd, speed, target, birdview, weight, semseg, depth)
                else:
                    replay_buffer.add_data(input_data, cmd, speed, target, birdview, weight)
        else:
            replay_buffer = torch.load(Path(config["log_dir"]) / "replay_buffer_episode_{}.saved".format(config["resume_episode"]))

        replay_buffer.normalized = True
        print("Replay buffer complete.")

        begin_episode = config['resume_episode']
        begin_episode += 1  # add one because we want to go to the next

        # overwrite the config to load the correct weights
        config['model_args']['image_ckpt'] = Path(config['log_dir']) / ('model-%d.th' % (begin_episode - 1))
    else:
        begin_episode = 0

    bzu.log.init(config['log_dir'], epoch=begin_episode * config['epoch_per_episode'])
    bzu.log.save_config(config)
    # teacher_config = bzu.log.load_config(config['teacher_args']['model_path'])

    criterion = LocationLoss()
    net = setup_image_model(**config["model_args"], device=config["device"], all_branch=True, imagenet_pretrained=False,
                            use_cv=use_cv, trained_cv=trained_cv)

    teacher_net = load_birdview_model(
        "resnet18",
        config['teacher_args']['model_path'],
        device=config['device'])

    image_agent_kwargs = {'camera_args': config["agent_args"]['camera_args']}

    coord_converter = CoordConverter(**config["agent_args"]['camera_args'])

    # optimizer = get_optimizer(net.parameters(), config["optimizer_args"]["lr"])

    for episode in tqdm.tqdm(range(begin_episode, config['max_episode'] + begin_episode), initial=begin_episode,
                             desc='Episode'):
        rollout(replay_buffer, coord_converter, net, teacher_net, episode, episode_length=config['episode_length'],
                image_agent_kwargs=image_agent_kwargs, port=config['port'], use_cv=use_cv)
        # import pdb; pdb.set_trace()
        _train(replay_buffer, net, teacher_net, criterion, coord_converter, bzu.log, config, episode)
        if buffer_type == "standard":
            print("Storing images, cmds, speeds and targets from replay buffer...")
            torch.save(replay_buffer.get_image_data(),
                           Path(config['log_dir']) / 'replay_buffer-{}_image_data.saved'.format(episode))
            print("Storing birdview images from replay buffer...")
            torch.save(replay_buffer.get_birdview_data(),
                       Path(config['log_dir']) / 'replay_buffer-{}_birdview_data.saved'.format(episode))
            print("Storing replay buffer weights ...")
            torch.save(replay_buffer.get_weights(),
                       Path(config["log_dir"]) / "replay_buffer-{}_weights.saved".format(episode))
            print("Replay buffer weights for episode number (", episode, ") saved.", sep="")
        else:
            print("Storing replay buffer from episode {}".format(episode))
            torch.save(replay_buffer, Path(config["log_dir"]) / "replay_buffer_episode_{}.saved".format(episode))
        print("Rollout-data stored.", sep="")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--log_iterations', default=100)
    parser.add_argument('--max_episode', type=int, default=20)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--epoch_per_episode', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--buffer_type', default="normal", choices=["standard", "disk"])
    parser.add_argument('--speed_noise', type=float, default=0.0)
    parser.add_argument('--batch_aug', type=int, default=1)
    parser.add_argument('--use_cv', default=False, action='store_true',
                        help="Use ground-truth computer vision (cv) images (semantic segmentation and depth estimation)")
    parser.add_argument('--trained_cv', default=False, action='store_true')

    # resume flag will continue from the chosen epoch with the saved replay buffer. Assumes identical config
    parser.add_argument('--resume_episode', type=int, default=-1)

    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--perception_ckpt', default="")

    # Teacher.
    parser.add_argument('--teacher_path', required=True)

    parser.add_argument('--fixed_offset', type=float, default=4.0)

    # Optimizer.
    parser.add_argument('--lr', type=float, default=1e-4)

    # Misc
    parser.add_argument('--port', type=int, default=2000)

    parsed = parser.parse_args()

    config = {
            'port': parsed.port,
            'log_dir': parsed.log_dir,
            'log_iterations': parsed.log_iterations,
            'batch_size': parsed.batch_size,
            'max_episode': parsed.max_episode,
            'episode_length' : parsed.episode_length,
            'speed_noise': parsed.speed_noise,
            'epoch_per_episode': parsed.epoch_per_episode,
            'device': 'cuda',
            'phase1_ckpt': parsed.ckpt,
            'resume_episode': parsed.resume_episode,
            'optimizer_args': {'lr': parsed.lr},
            'buffer_args': {
                'buffer_limit': 200000,
                'batch_aug': parsed.batch_aug,
                'augment': 'super_hard',
                'aug_fix_iter': 819200,
                'type' : parsed.buffer_type,
            },
            'model_args': {
                'model': 'image_ss',
                'image_ckpt' : parsed.ckpt,
                'backbone': BACKBONE,
                'perception_ckpt': parsed.perception_ckpt,
                'input_channel': len(DEFAULT_CLASSES) + 5 if parsed.use_cv else 3
                },
            'agent_args': {
                'camera_args': {
                    'w': 384,
                    'h': 160,
                    'fov': 90,
                    'world_y': 1.4,
                    'fixed_offset': parsed.fixed_offset,
                },
                'use_cv': parsed.use_cv
            },
            'teacher_args' : {
                'model_path': parsed.teacher_path,
            }
        }

    train(config)
