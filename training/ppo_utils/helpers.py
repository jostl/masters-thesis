import cv2
import numpy as np

import bird_view.utils.bz_utils as bzu
import bird_view.utils.carla_utils as cu
from bird_view.models.common import crop_birdview
from benchmark.goal_suite import PointGoalSuite


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


def get_reward(goal_suite: PointGoalSuite, speed, alpha=1, beta=1, phi=250, delta=250):
    # Reward function from Affordance-based Reinforcement Learning for Urban Driving
    # https://arxiv.org/pdf/2101.05970.pdf
    def infraction_penalty():
        if goal_suite.collided or goal_suite.traffic_tracker.ran_light:
            return -phi * speed - delta
        return 0

    def distance_penalty():
        next_waypoint_location = goal_suite._next  # Get the location of the next waypoint
        player_location = goal_suite._player.get_location()
        distance = player_location.distance(next_waypoint_location)  # Get distance between actor and next waypoint
        return -beta * distance

    def speed_reward():
        return alpha * speed

    return speed_reward() + distance_penalty() + infraction_penalty()


def _paint(observations, control, diagnostic, reward, debug, env):
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    CROP_SIZE = 192
    X = 176
    Y = 192 // 2
    R = 2

    birdview = cu.visualize_birdview(observations['birdview'])
    birdview = crop_birdview(birdview)

    if 'big_cam' in observations:
        canvas = np.uint8(observations['big_cam']).copy()
        rgb = np.uint8(observations['rgb']).copy()
    else:
        canvas = np.uint8(observations['rgb']).copy()

    def _stick_together(a, b, axis=1):

        if axis == 1:
            h = min(a.shape[0], b.shape[0])

            r1 = h / a.shape[0]
            r2 = h / b.shape[0]

            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))

            return np.concatenate([a, b], 1)

        else:
            h = min(a.shape[1], b.shape[1])

            r1 = h / a.shape[1]
            r2 = h / b.shape[1]

            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))

            return np.concatenate([a, b], 0)

    def _write(text, i, j, canvas=canvas, fontsize=0.4):
        rows = [x * (canvas.shape[0] // 10) for x in range(10 + 1)]
        cols = [x * (canvas.shape[1] // 9) for x in range(9 + 1)]
        cv2.putText(
            canvas, text, (cols[j], rows[i]),
            cv2.FONT_HERSHEY_SIMPLEX, fontsize, WHITE, 1)

    _command = {
        1: 'LEFT',
        2: 'RIGHT',
        3: 'STRAIGHT',
        4: 'FOLLOW',
    }.get(observations['command'], '???')

    _reward = round(reward, 2)

    if 'big_cam' in observations:
        fontsize = 0.8
    else:
        fontsize = 0.4

    _write('Command: ' + _command, 1, 0, fontsize=fontsize)
    _write('Velocity: %.1f' % np.linalg.norm(observations['velocity']), 2, 0, fontsize=fontsize)
    _write('Reward: %.1f' % _reward, 3, 0, fontsize=fontsize)
    _write('Steer: %.2f' % control.steer, 4, 0, fontsize=fontsize)
    _write('Throttle: %.2f' % control.throttle, 5, 0, fontsize=fontsize)
    _write('Brake: %.1f' % control.brake, 6, 0, fontsize=fontsize)

    _write('Collided: %s' % diagnostic['collided'], 1, 6, fontsize=fontsize)
    _write('Invaded: %s' % diagnostic['invaded'], 2, 6, fontsize=fontsize)
    _write('Lights Ran: %d/%d' % (env.traffic_tracker.total_lights_ran, env.traffic_tracker.total_lights), 3, 6,
           fontsize=fontsize)
    _write('Goal: %.1f' % diagnostic['distance_to_goal'], 4, 6, fontsize=fontsize)

    _write('Time: %d' % env._tick, 5, 6, fontsize=fontsize)
    _write('FPS: %.2f' % (env._tick / (diagnostic['wall'])), 6, 6, fontsize=fontsize)

    for x, y in debug.get('locations', []):
        x = int(X - x / 2.0 * CROP_SIZE)
        y = int(Y + y / 2.0 * CROP_SIZE)

        S = R // 2
        birdview[x - S:x + S + 1, y - S:y + S + 1] = RED

    for x, y in debug.get('locations_world', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)

        S = R // 2
        birdview[x - S:x + S + 1, y - S:y + S + 1] = RED

    for x, y in debug.get('locations_birdview', []):
        S = R // 2
        birdview[x - S:x + S + 1, y - S:y + S + 1] = RED

    for x, y in debug.get('locations_pixel', []):
        S = R // 2
        if 'big_cam' in observations:
            rgb[y - S:y + S + 1, x - S:x + S + 1] = RED
        else:
            canvas[y - S:y + S + 1, x - S:x + S + 1] = RED

    for x, y in debug.get('curve', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)

        try:
            birdview[x, y] = [155, 0, 155]
        except:
            pass

    if 'target' in debug:
        x, y = debug['target'][:2]
        x = int(X - x * 4)
        y = int(Y + y * 4)
        birdview[x - R:x + R + 1, y - R:y + R + 1] = [0, 155, 155]

    ox, oy = observations['orientation']
    rot = np.array([
        [ox, oy],
        [-oy, ox]])
    u = observations['node'] - observations['position'][:2]
    v = observations['next'] - observations['position'][:2]
    u = rot.dot(u)
    x, y = u
    x = int(X - x * 4)
    y = int(Y + y * 4)
    v = rot.dot(v)
    x, y = v
    x = int(X - x * 4)
    y = int(Y + y * 4)

    if 'big_cam' in observations:
        _write('Network input/output', 1, 0, canvas=rgb)
        _write('Projected output', 1, 0, canvas=birdview)
        full = _stick_together(rgb, birdview)
    else:
        full = _stick_together(canvas, birdview)

    if 'image' in debug:
        full = _stick_together(full, cu.visualize_predicted_birdview(debug['image'], 0.01))

    if 'big_cam' in observations:
        full = _stick_together(canvas, full, axis=0)

    bzu.show_image('canvas', full)
