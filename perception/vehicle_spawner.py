#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import math
import random
from typing import List

import glob
import os
import sys
from typing import Dict

import carla


class VehicleSpawner(object):

    def __init__(self, client: carla.Client, world: carla.World, safe_mode=True):
        self.client = client
        self.world = world
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        self.blueprintsWalkers = world.get_blueprint_library().filter("walker.pedestrian.*")
        self.vehicles_list: List[int] = []
        self.walkers_list = []
        self.all_id = []
        self.all_actors = []
        self.safe_mode = safe_mode
        self._bad_colors = [
            "255,255,255", "183,187,162", "237,237,237",
            "134,134,134", "243,243,243", "127,130,135",
            "109,109,109", "181,181,181", "140,140,140",
            "181,178,124", "171,255,0", "251,241,176",
            "158,149,129", "233,216,168", "233,216,168",
            "108,109,126", "193,193,193", "227,227,227",
            "151,150,125", "206,206,206", "255,222,218",
            "211,211,211", "191,191,191"
        ] if safe_mode else []

    def init_traffic_manager(self):
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.global_percentage_speed_difference(25.0)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_synchronous_mode(True)

    def spawn_nearby(self, hero_spawn_point_index, number_of_vehicles_min, number_of_vehicles_max,
                     number_of_walkers_min, number_of_walkers_max, radius):

        number_of_vehicles = random.randint(number_of_vehicles_min, number_of_vehicles_max)
        number_of_walkers = random.randint(number_of_walkers_min, number_of_walkers_max)
        print(f"Attempting to spawn {number_of_vehicles} vehicles, {number_of_walkers} walkers")
        valid_spawn_points = self.get_valid_spawn_points(hero_spawn_point_index, radius)

        if self.safe_mode:
            self.blueprints = [x for x in self.blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            self.blueprints = [x for x in self.blueprints if not x.id.endswith('isetta')]
            self.blueprints = [x for x in self.blueprints if not x.id.endswith('carlacola')]
            self.blueprints = [x for x in self.blueprints if not x.id.endswith('cybertruck')]
            self.blueprints = [x for x in self.blueprints if not x.id.endswith('t2')]
            self.blueprints = [x for x in self.blueprints if not x.id.endswith('coupe')]

        number_of_spawn_points = len(valid_spawn_points)

        if number_of_spawn_points > number_of_vehicles:
            random.shuffle(valid_spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(valid_spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(self.blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                while color in self._bad_colors:
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                print(f"Vehicle spawn error: {response.error}")
            else:
                self.vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0  # how many pedestrians will run
        percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(self.blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                print(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        # tick to ensure client receives the last transform of the walkers we have just created
        self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        print(f'Spawned {len(self.vehicles_list):d} vehicles and {len(self.walkers_list):d} walkers,')

    def get_valid_spawn_points(self, hero_spawn_point_index, radius):
        hero_spawn_point = self.spawn_points[hero_spawn_point_index]
        hero_x = hero_spawn_point.location.x
        hero_y = hero_spawn_point.location.y
        valid_spawn_points = []
        for spawn_point in self.spawn_points:
            # Distance between spaw points
            loc = hero_spawn_point.location
            dx = spawn_point.location.x - loc.x
            dy = spawn_point.location.y - loc.y
            distance = math.sqrt(dx * dx + dy * dy)
            min_distance = 10
            if spawn_point == hero_spawn_point or distance < min_distance:
                continue
            if radius != 0:
                x = spawn_point.location.x
                y = spawn_point.location.y
                yaw = spawn_point.rotation.yaw
                angle_diff = hero_spawn_point.rotation.yaw - yaw
                angle_diff = abs((angle_diff + 180) % 360 - 180)

                if abs(hero_x - x) <= radius and abs(hero_y - y) <= radius and angle_diff < 50:
                    valid_spawn_points.append(spawn_point)
            else:
                valid_spawn_points.append(spawn_point)
        return valid_spawn_points

    def destroy_vehicles(self):
        print(f'Destroying {len(self.vehicles_list):d} vehicles.\n')
        self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.vehicles_list], True)
        self.vehicles_list.clear()

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('\ndestroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.all_id], True)
        self.walkers_list = []
        self.all_id = []
        self.all_actors = []
