# %%===========================================================================
# Import packages
import os
import cv2
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil

from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

from utils  import *

plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Define IO parameters
CONFIG_PATH = 'config'

EXPERIMENTS = [
    '10-microns particles-60X',
    'Isolada 3--3',
    'Isolada 3--2',
    'Isolada-2-10 um',
    'Calibration2_Single Cell',
    'Calibration1_Single Cell',
    'Four-mixing particles together',
    'Several 10-micron-particles together',
    'Calibration 10-microns',
    '30 microns-beads-60X-measuring 2',
    'Calibration-1-4 Cells',
    '3 particles_10 um',
]

# for EXPERIMENT in EXPERIMENTS:
EXPERIMENT = EXPERIMENTS[0]

# Read configuration parameters from JSON file
CONFIG_FILENAME = os.path.join(CONFIG_PATH, EXPERIMENT + '.json')
with open(CONFIG_FILENAME, 'r') as fp:
    config = json.load(fp)