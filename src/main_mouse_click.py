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















#%%
image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

p_upper_left  = {'x':30 , 'y': 10}
p_lower_right = {'x':320, 'y':280}

e = (
    (
        int((p_upper_left['x']  + p_lower_right['x'])/2),
        int((p_upper_left['y']  + p_lower_right['y'])/2)
    ),
    (
        int((p_lower_right['x'] - p_upper_left['x'])/2),
        int((p_lower_right['y'] - p_upper_left['y'])/2)
    ),
    0
)
init_level_set = np.zeros(image.shape, image.dtype)
init_level_set = cv2.ellipse(init_level_set, e[0], e[1], e[2], 0, 360, 255, 1)

while True:
    
    cv2.imshow('image', image)
    cv2.imshow('init_level_set', init_level_set)

    key = 0xFF & cv2.waitKey(1)

    if key == ord('q'):
        break

cv2.destroyAllWindows()

#%%
def my_imshow(image):
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyWindow('image')
    
image = 1.0 - normalize_image(image)
evolution = compute_snakes(
    image, threshold=config["SNAKES_THRESH"], 
    n_iter=config["SNAKES_N_ITER"], init_level_set=init_level_set)
evolution = np.array(evolution)








#%%
my_imshow(evolution[-1])








#%%
name_list = [os.path.join(config['OUTPUT_PATH'], 'split', 'split_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
images = read_list_of_images(name_list)
show_list_of_images(images)

# %%
global n_clicks, mouseX, mouseY, image_1, image_2

# ind = 0
ind = 10
# ind = config['N_IMAGES'] - 10

image    = images[ind].copy()
image_1  = image.copy()
image_2  = image.copy()
n_clicks = int(0)

posList = []
def draw_circle(event, x, y, flags, param):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print('x = %d, y = %d'%(x, y))
        posList.append((x, y))
        cv2.drawMarker(image_1, (x,y), (255,0,0), cv2.MARKER_CROSS, 10, 2)
        global n_clicks
        n_clicks = n_clicks + 1
        mouseX, mouseY = x, y

cv2.namedWindow('image_1', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('image_2', cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('image_1', draw_circle)

histogram = draw_histogram(image)

while(True):

    cv2.imshow('image_1'  , image_1)
    # cv2.imshow('image_2'  , image_2)
    # cv2.imshow('histogram', histogram)

    key = 0xFF & cv2.waitKey(20)

    if key == ord('q'):
        break

    elif n_clicks == 5:

        posNp = np.array(posList)
        ellipse = fit_ellipses([posNp])
        image_2 = draw_ellipses(image_2, ellipse)

        init_level_set = np.zeros(image.shape, dtype=np.int8)
        init_level_set = draw_ellipses(image_2, ellipse)
        # image = 1.0 - normalize_image(image)
        # evolution = compute_snakes(image, threshold=config["SNAKES_THRESH"], n_iter=config["SNAKES_N_ITER"])
        # evolution = np.array(evolution)


        # #     # Plot intermediate SNAKES iterations
        # fig = plt.figure(figsize=(10,5))
        # plt.imshow(image, cmap="gray")
        # iterations = np.linspace(0, config["SNAKES_N_ITER"], config["SNAKES_N_CONTOURS"]).astype(int)
        # # colors = ['k', 'y', 'g', 'c', 'm', 'r']
        # plt.imshow(evolution[-1], cmap='gray')
        # title = "Morphological GAC evolution"
        # # ax.contour(evolution[-1], [0.5], colors='r')
        # # ax.legend(bbox_to_anchor=(1.04,0.7), loc="upper left")
        # fig.tight_layout()
        # plt.gca().set_aspect('equal')
        # plt.gca().set_axis_off()
        # # plt.show()
    
cv2.destroyAllWindows()

#%%
mu, sigma = return_image_statistics(images[0], mask_shape = (3,3))

#%%
cv2.imshow('image', image)
cv2.imshow('mu'   , mu)
cv2.imshow('sigma', sigma)
cv2.waitKey(0)
cv2.destroyAllWindows()