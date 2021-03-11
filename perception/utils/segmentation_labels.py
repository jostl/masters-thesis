# Original file:
# https://github.com/AudunWA/master-thesis-code/blob/master/prediction_pipeline/generate_segmentation_files.py
# Modified by Martin Hermansen and Jostein Lilleløkken to fit our needs for our specialization project
# Many of the changes are inspired by and/or based on the work by
# Arbo&Dalen (2020)

import numpy as np

# Carla colors
carla_road = np.array([np.array([128, 64, 128], dtype='uint8')])
carla_sidewalk = np.array([np.array([244, 35, 232], dtype='uint8')])
carla_lane_markings_colors = np.array([np.array([157, 234, 50], dtype='uint8')])
carla_vehicle_colors = np.array([np.array([0, 0, 142], dtype='uint8')])
carla_humans_colors = np.array([np.array([220, 20, 60], dtype='uint8')])

#CARLA_CLASSES = np.array(
#    [carla_road, carla_sidewalk, carla_lane_markings_colors, carla_humans_colors, carla_vehicle_colors])


CARLA_CLASSES = {0: ("Unlabeled", (0, 0, 0)),
                 1: ("Building", (70, 70, 70)),
                 2: ("Fence", (100, 40, 40)),
                 3: ("Other", (55, 90, 80)),
                 4: ("Pedestrian", (220, 20, 60)),
                 5: ("Pole", (153, 153, 153)),
                 6: ("RoadLine", (157, 234, 50)),
                 7: ("Road", (128, 64, 128)),
                 8: ("SideWalk", (244, 35, 232)),
                 9: ("Vegetation", (107, 142, 35)),
                 10: ("Vehicles", (0, 0, 142)),
                 11: ("Wall", (102, 102, 156)),
                 12: ("TrafficSign", (220, 220, 0)),
                 13: ("Sky", (70, 130, 180)),
                 14: ("Ground", (81, 0, 81)),
                 15: ("Bridge", (150, 100, 100)),
                 16: ("RailTrack", (230, 150, 140)),
                 17: ("GuardRail", (180, 165, 180)),
                 18: ("TrafficLight", (250, 170, 30)),
                 19: ("Static", (110, 190, 160)),
                 20: ("Dynamic", (170, 120, 50)),
                 21: ("Water", (45, 60, 150)),
                 22: ("Terrain", (145, 170, 100))}
