# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:49:18 2021

@author: Vieira
"""

#%%
import os
import json 
import numpy as np

N_IMAGES = 18

CIRCLES_GT = np.zeros((N_IMAGES, 3), dtype=int)
CIRCLES_GT[0, :] = np.array([452, 322, 47])
CIRCLES_GT[0, :] = np.array([453, 332, 47])

config = {
    'CROP_RECTANGLE'                : (816, 640, 816, 640),
    'GRAD_WSIZE'                    : 31,
    'INPUT_FILENAME'                : os.path.join(os.path.join("..", "dataset", "Calibration 10-microns"), 
                                                                "Calibration 10-microns.tif"),
    'INPUT_PATH'                    : os.path.join("..", "dataset", "Calibration 10-microns"),
    'MEDBLUR_WSIZE'                 : 23,
    'METERS_PER_PIXEL_RATIO'        : None,
    'N_IMAGES'                      : 18,
    'OUTPUT_PATH'                   : os.path.join("..", "results", "Calibration 10-microns"),
    'SNAKES_N_CONTOURS'             : 6,
    'SNAKES_N_ITER'                 : 500,
    'SNAKES_THRESH'                 : 0.83,
    'SQUARED_METERS_PER_PIXEL_RATIO': None,
    }

with open('config_2.json', 'w') as fp:
    json.dump(config, fp, sort_keys=True, indent=4)

np.save("config_2.npy", CIRCLES_GT)