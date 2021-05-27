# 1. Set up carla client
# 2. Spawn a lot of vehicles and pedestrians
# 3. Loop:
# - 1. Pick a random actor
# - 2. Attach rgb, depth, and semantic sensor
# - 3. Set random camera attributes (fov, pitch, height)
# - 4. Save sensor output
# 4. Choose next map and go to 2
import glob
import math
import os
import random
import sys
from queue import Queue, Empty
from typing import Dict, List, Tuple
import numpy as np
import tqdm

try:
    sys.path.append(glob.glob('PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("could not find the CARLA egg")
    pass
import carla

from perception.vehicle_spawner import VehicleSpawner
from utils.carla_utils import PRESET_WEATHERS

# We want to get a deterministic output

FPS = 10
TIMEOUT = 2
SENSOR_TICK = 40  # Seconds between each sensor tick

TRAINING_WEATHERS = [1, 3, 6, 8]
TEST_WEATHERS = [4, 10, 14]
ALL_WEATHERS = TRAINING_WEATHERS + TEST_WEATHERS


def spawn_sensors(world: carla.World, actor: carla.Actor):
    sensors = {}
    blueprints = {
        "rgb": world.get_blueprint_library().find('sensor.camera.rgb'),
        "segmentation": world.get_blueprint_library().find('sensor.camera.semantic_segmentation'),
        "depth": world.get_blueprint_library().find('sensor.camera.depth')
    }
    bad_vehicles = ["vehicle.carlamotors.carlacola", "vehicle.tesla.cybertruck",
                    "vehicle.chargercop2020.chargercop2020"]

    # the camera placement is very weird on this car
    if actor.type_id in bad_vehicles or actor.type_id.endswith("isetta"):
        return False, sensors

    fov = 90
    # yaw = random.gauss(0, 30)
    sensor_transform = carla.Transform(carla.Location(1, 0, 1.4), carla.Rotation(0, 0, 0))
    for sensor_name, blueprint in blueprints.items():
        blueprint.set_attribute("image_size_x", "384")
        blueprint.set_attribute("image_size_y", "160")
        blueprint.set_attribute("fov", str(fov))
        blueprint.set_attribute("sensor_tick", str(SENSOR_TICK))  # Get new data every x second
        sensor: carla.Sensor = world.spawn_actor(blueprint, sensor_transform, attach_to=actor)
        sensors[sensor_name] = sensor
    return True, sensors


def retrieve_data(frame, sensor_queue, timeout) -> carla.Image:
    while True:
        data = sensor_queue.get(timeout=timeout)
        if data.frame == frame:
            return data
        elif data.frame == frame - 1:
            # print(f"Got one-off frame (expected {frame}, got {data.frame})")
            return data
        else:
            # print(f"Expected {frame}, got {data.frame}")
            return data


def get_color_converter(sensor_name):
    if sensor_name == "depth":
        return carla.ColorConverter.LogarithmicDepth
    else:
        return carla.ColorConverter.Raw


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


def spawn_vehicles(client, world, n_vehicles, n_data_collection_vehicles=35):
    SpawnActor = carla.command.SpawnActor
    SetAutoPilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    blueprints = world.get_blueprint_library().filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    batch = []

    for i in range(n_vehicles - n_data_collection_vehicles):
        blueprint = np.random.choice(blueprints)
        blueprint.set_attribute('role_name', 'autopilot')

        if blueprint.has_attribute('color'):
            color = np.random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        if blueprint.has_attribute('driver_id'):
            driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)

        spawn_point = spawn_points.pop(random.randrange(len(spawn_points)))
        spawn_command = SpawnActor(blueprint, spawn_point).then(SetAutoPilot(FutureActor, True))
        batch.append(spawn_command)
    data_collection_batch = []
    good_vehicles = ["vehicle.mustang.mustang", "vehicle.lincoln2020.mkz2020",
                     "vehicle.chevrolet.impala", "vehicle.audi.tt", "vehicle.mini.cooperst"]
    for i in range(n_data_collection_vehicles):
        blueprint = blueprints.find(random.choice(good_vehicles))
        # blueprint = blueprints.find("vehicle.mustang.mustang")
        blueprint.set_attribute('role_name', 'autopilot')

        if blueprint.has_attribute('color'):
            color = np.random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        if blueprint.has_attribute('driver_id'):
            driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        spawn_point = spawn_points.pop(random.randrange(len(spawn_points)))
        spawn_command = SpawnActor(blueprint, spawn_point).then(SetAutoPilot(FutureActor, True))
        data_collection_batch.append(spawn_command)

    responses = client.apply_batch_sync(batch, True)
    actor_ids = []
    for response in responses:
        actor_ids.append(response.actor_id)

    data_collection_ids = []
    responses = client.apply_batch_sync(data_collection_batch, True)
    for response in responses:
        actor_ids.append(response.actor_id)
        data_collection_ids.append(response.actor_id)
    print("spawned %d vehicles" % len(actor_ids))
    return actor_ids, data_collection_ids


def spawn_pedestrians(client, world, n_pedestrians):
    SpawnActor = carla.command.SpawnActor

    peds_spawned = 0

    walkers = []
    controllers = []

    while peds_spawned < n_pedestrians:
        spawn_points = []
        _walkers = []
        _controllers = []

        for i in range(n_pedestrians - peds_spawned):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()

            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprints)

            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')

            batch.append(SpawnActor(walker_bp, spawn_point))

        for result in client.apply_batch_sync(batch, True):
            if result.error:
                print(result.error)
            else:
                peds_spawned += 1
                _walkers.append(result.actor_id)

        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        batch = [SpawnActor(walker_controller_bp, carla.Transform(), walker) for walker in _walkers]

        for result in client.apply_batch_sync(batch, True):
            if result.error:
                print(result.error)
            else:
                _controllers.append(result.actor_id)

        controllers.extend(_controllers)
        walkers.extend(_walkers)

    for controller in world.get_actors(controllers):
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(1 + random.random())

    print("spawned %d pedestrians" % len(controllers))

    return world.get_actors(walkers), world.get_actors(controllers)


weather_names = {
    1: 'clear_noon',
    3: 'wet_noon',
    4: 'wet_cloudy_noon',
    6: 'hardrain_noon',
    8: 'clear_sunset',
    10: "after_rain_sunset",
    14: "soft_rain_sunset"
}


def main():
    town = "Town01"
    weathers = TRAINING_WEATHERS
    # weathers = TEST_WEATHERS
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    traffic_manager: carla.TrafficManager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    traffic_manager.global_percentage_speed_difference(25.0)

    folder_name = "train_new_height"

    n_vehicles = 100
    n_pedestrians = 250
    total_images = 20000
    n_images_per_weather = total_images // len(weathers)

    print(f"Loading {town}")
    client.load_world(town)
    world: carla.World = client.get_world()
    start_frame = world.apply_settings(carla.WorldSettings(
        no_rendering_mode=False,
        synchronous_mode=True,
        fixed_delta_seconds=1 / FPS))

    for weather in tqdm.tqdm(weathers):
        weather_name = weather_names[weather]
        progress = tqdm.tqdm(range(n_images_per_weather), desc=weather_name)

        world.set_weather(PRESET_WEATHERS[weather])
        vehicles_list, data_collection_list = spawn_vehicles(client, world, n_vehicles)
        walkers, walker_controllers = spawn_pedestrians(client, world, n_pedestrians)

        # print("Running the world for 5 seconds before capturing...")
        frame = start_frame
        while frame < start_frame + FPS * 5:
            frame = world.tick()
        # print("Starting capture!")

        all_sensors: List[carla.Sensor] = []
        vehicle_sensor_queues: Dict[int, Dict[str, Tuple[Queue, carla.Sensor]]] = {}

        n_spawned_sensor = 0
        for vehicle_id in data_collection_list:
            vehicle = world.get_actor(vehicle_id)
            success, sensors = spawn_sensors(world, vehicle)
            if success:
                n_spawned_sensor += 1
                for sensor_name, sensor in sensors.items():
                    all_sensors.append(sensor)
                    queue = Queue()
                    sensor.listen(queue.put)
                    if vehicle_id not in vehicle_sensor_queues:
                        vehicle_sensor_queues[vehicle_id] = {}
                    vehicle_sensor_queues[vehicle_id][sensor_name] = (queue, sensor)

        while frame < start_frame + FPS * 5:
            frame = world.tick()
        n_images_per_vehicle_per_weather = n_images_per_weather // n_spawned_sensor

        for i in range(0, n_images_per_vehicle_per_weather):
            for i in range(FPS * 5):
                frame = world.tick()
            # print(f"Tick ({frame})")

            for vehicle_id, vehicle_sensors in vehicle_sensor_queues.items():
                frame_nums = []

                changed_yaw = False

                for sensor_name, (queue, sensor) in vehicle_sensors.items():
                    try:
                        image = retrieve_data(frame, queue, TIMEOUT)
                        if sensor_name == "rgb":
                            progress.update(1)
                        # print(f"Got image from {sensor_name} {image.frame}")
                        frame_nums.append(image.frame)
                        image.save_to_disk(
                            f"data/perception/{folder_name}/{sensor_name}/{weather_name}_{vehicle_id}_{image.frame}.png",
                            get_color_converter(sensor_name))

                        if not changed_yaw:
                            old_yaw = sensor.get_transform().rotation.yaw
                            new_yaw = random.gauss(0, 30)
                            while abs(new_yaw - old_yaw) < 30:
                                new_yaw = random.gauss(0, 30)

                            changed_yaw = True

                        sensor.set_transform(carla.Transform(carla.Location(1, 0, 1.4), carla.Rotation(0, 0, 0)))
                    except Empty:
                        pass
                        # print("Empty queue")
                assert all(frame == frame_nums[0] for frame in frame_nums)

            # Run amount of ticks equal to the sensors' sensor_tick value
            for t in range(0, int(SENSOR_TICK * FPS)):
                frame = world.tick()
                # print(f"Tick ({frame})")

        # Clean up
        # traffic_manager.set_synchronous_mode(False)
        for sensor in all_sensors:
            sensor.destroy()
        # print(f'Destroying {len(vehicles_list):d} vehicles.\n')
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in vehicles_list], True)

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(len(walker_controllers)):
            walker_controllers[i].stop()

        # print('\ndestroying %d walkers' % len(walkers))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in (walkers)], True)
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in (walker_controllers)], True)


if __name__ == "__main__":
    main()
