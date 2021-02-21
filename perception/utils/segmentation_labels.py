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

CARLA_CLASSES = np.array(
    [carla_road, carla_sidewalk, carla_lane_markings_colors, carla_humans_colors, carla_vehicle_colors])
