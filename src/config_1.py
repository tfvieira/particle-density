# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:42:44 2021

@author: Vieira
"""
import json

config = {
    'CROP_RECTANGLE': (1000, 600, 1200, 600),
    'GRAD_WSIZE': 31,
    'INPUT_FILENAME': '..\\dataset\\10-microns particles-60X\\10-microns particles-60X.tif',
    'INPUT_PATH': '..\\dataset\\10-microns particles-60X',
    'MEDBLUR_WSIZE': 23,
    'METERS_PER_PIXEL_RATIO': 1.1363636363636364e-07,
    'N_IMAGES': 20,
    'OUTPUT_PATH': '..\\results\\10-microns particles-60X',
    'SNAKES_N_CONTOURS': 6,
    'SNAKES_N_ITER': 500,
    'SNAKES_THRESH': 0.83,
    'SQUARED_METERS_PER_PIXEL_RATIO': 6.456611570247935e-10,
    'CIRCLES_GT' : None
    }

with open('config_1.json', 'w') as fp:
    json.dump(config, fp, sort_keys=True, indent=4)