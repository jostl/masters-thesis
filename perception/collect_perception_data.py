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

# We want to get a deterministic output
# random.seed(1)

FPS = 10
NUMBER_OF_IMAGES_PER_VEHICLE_PER_WORLD = 100
TIMEOUT = 2
SENSOR_TICK = 5.0  # Seconds between each sensor tick


def spawn_sensors(world: carla.World, actor: carla.Actor) -> Dict[str, carla.Sensor]:
    sensors = {}
    blueprints = {
        "rgb": world.get_blueprint_library().find('sensor.camera.rgb'),
        "segmentation": world.get_blueprint_library().find('sensor.camera.semantic_segmentation'),
        "depth": world.get_blueprint_library().find('sensor.camera.depth')
    }

    fov = 90
    yaw = random.gauss(0, 45)
    sensor_transform = carla.Transform(carla.Location(0, 0, 3), carla.Rotation(0, yaw, 0))
    for sensor_name, blueprint in blueprints.items():
        blueprint.set_attribute("image_size_x", "640")
        blueprint.set_attribute("image_size_y", "480")
        blueprint.set_attribute("fov", str(fov))
        blueprint.set_attribute("sensor_tick", str(SENSOR_TICK))  # Get new data every x second
        sensor: carla.Sensor = world.spawn_actor(blueprint, sensor_transform, attach_to=actor)
        sensors[sensor_name] = sensor
    return sensors


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
    elif sensor_name == "segmentation":
        return carla.ColorConverter.CityScapesPalette
    else:
        return carla.ColorConverter.Raw


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self._storm = Storm(weather.precipitation)

    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)


def main():
    worlds = ["Town02"]
    # worlds = ["Town01", "Town02", "Town03", "Town04", "Town05"]
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    traffic_manager: carla.TrafficManager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    traffic_manager.global_percentage_speed_difference(25.0)
    # traffic_manager.set_hybrid_physics_mode(True)
    # traffic_manager.set_synchronous_mode(True)

    for world_name in worlds:
        print(f"Loading {world_name}")
        client.load_world(world_name)
        world: carla.World = client.get_world()

        start_frame = world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=1 / FPS))
        weather = Weather(carla.WeatherParameters.ClearNoon)
        world.set_weather(weather.weather)

        vehicle_spawner = VehicleSpawner(client, world, False)
        vehicle_spawner.spawn_nearby(0, 40, 40, 40, 80, 1000000)

        print("Running the world for 5 seconds before capturing...")
        frame = start_frame
        while frame < start_frame + FPS * 5:
            frame = world.tick()
        print("Starting capture!")

        all_sensors: List[carla.Sensor] = []
        vehicle_sensor_queues: Dict[int, Dict[str, Tuple[Queue, carla.Sensor]]] = {}
        for vehicle_id in vehicle_spawner.vehicles_list:
            vehicle = world.get_actor(vehicle_id)
            sensors = spawn_sensors(world, vehicle)

            for sensor_name, sensor in sensors.items():
                all_sensors.append(sensor)
                queue = Queue()
                sensor.listen(queue.put)
                if vehicle_id not in vehicle_sensor_queues:
                    vehicle_sensor_queues[vehicle_id] = {}
                vehicle_sensor_queues[vehicle_id][sensor_name] = (queue, sensor)

        while frame < start_frame + FPS * 5:
            frame = world.tick()

        for i in range(0, NUMBER_OF_IMAGES_PER_VEHICLE_PER_WORLD):
            weather.tick(1)
            world.set_weather(weather.weather)
            frame = world.tick()
            print(f"Tick ({frame})")

            for vehicle_id, vehicle_sensors in vehicle_sensor_queues.items():
                new_yaw = random.gauss(0, 45)
                frame_nums = []
                for sensor_name, (queue, sensor) in vehicle_sensors.items():
                    try:
                        image = retrieve_data(frame, queue, TIMEOUT)
                        print(f"Got image from {sensor_name} {image.frame}")
                        frame_nums.append(image.frame)
                        image.save_to_disk(f"data/perception/carla_test2/{sensor_name}/{vehicle_id}_{image.frame}.png",
                                           get_color_converter(sensor_name))
                        sensor.set_transform(carla.Transform(carla.Location(0, 0, 3), carla.Rotation(0, new_yaw, 0)))
                    except Empty:
                        print("Empty queue")
                assert all(frame == frame_nums[0] for frame in frame_nums)

            # Run amount of ticks equal to the sensors' sensor_tick value
            settings = world.get_settings()
            settings.no_rendering_mode = True
            world.apply_settings(settings)
            for t in range(0, int(SENSOR_TICK * FPS) - 8):
                weather.tick(1 / FPS * 5)
                world.set_weather(weather.weather)
                frame = world.tick()
                print(f"Tick ({frame})")
            settings.no_rendering_mode = False
            world.apply_settings(settings)

            # Do rest of ticks with rendering (to hopefully get rain effect)
            for i in range(0, 8):
                weather.tick(1 / FPS * 5)
                world.set_weather(weather.weather)
                frame = world.tick()
                print(f"Tick ({frame})")

        # Clean up
        # traffic_manager.set_synchronous_mode(False)
        for sensor in all_sensors:
            sensor.destroy()
        vehicle_spawner.destroy_vehicles()


main()
