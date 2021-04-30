# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 20:52:55 2021

@author: Vieira
"""

# %%===========================================================================
# Import packages
import os
import cv2
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils  import *#read_list_of_images
from snakes import *
from detect_blur_fft import detect_blur_fft

plt.style.use('seaborn')
plt.rcParams["figure.dpi"] = 300
plt.rcParams['savefig.dpi'] = 300

# %%===========================================================================
# Define IO parameters -- 10-microns particles-60X

# CONFIG_FILENAME = "config_1.json"
# CONFIG_FILENAME = "config_2.json"
CONFIG_FILENAME = "config_3.json"

# CIRCLES_GT_FILENAME = "config_2.npy"

# CIRCLES_GT = np.load(CIRCLES_GT_FILENAME)
with open(CONFIG_FILENAME, 'r') as fp:
    config = json.load(fp)

# %%===========================================================================
# Split one TIF image into many images, each corresponding to one TIF layer
split_images(config["INPUT_FILENAME"], 
              config["OUTPUT_PATH"])

# %%===========================================================================
# Crop the images to contain only the particles
crop_images(config["INPUT_FILENAME"], 
            config["OUTPUT_PATH"], 
            rectangle=config["CROP_RECTANGLE"])

# %%===========================================================================
# Pre-process all images
name_list = [os.path.join(config["OUTPUT_PATH"], "crop_" + str(x) + ".tif") for x in range(config["N_IMAGES"])]
images = read_list_of_images(name_list)
preprocessed_images = preprocess_list_of_images(images)
preprocessed_filenames = [os.path.join(config["OUTPUT_PATH"], "preprocessed_" + str(x) + ".tif") for x in range(config["N_IMAGES"])]
write_list_of_images(preprocessed_filenames, preprocessed_images)

# %%===========================================================================
# Show list of pre-processed images
# preprocessed_filenames = [os.path.join(config["OUTPUT_PATH"], "preprocessed_" + str(x) + ".tif") for x in range(config["N_IMAGES"])]
# preprocessed_images = read_list_of_images(preprocessed_filenames)
# show_list_of_images(preprocessed_images)

# %%===========================================================================
# Process Ground truths
name_list = [os.path.join(config["OUTPUT_PATH"], "split_" + str(x) + ".tif") for x in range(config["N_IMAGES"])]
gt_list   = [os.path.join(config["OUTPUT_PATH"], "gt_" + str(x) + ".tif") for x in range(config["N_IMAGES"])]
out_list  = [os.path.join(config["OUTPUT_PATH"], "split_gt_" + str(x) + ".tif") for x in range(config["N_IMAGES"])]
mom_list  = [os.path.join(config["OUTPUT_PATH"], "moments_" + str(x) + ".json") for x in range(config["N_IMAGES"])]
areas, moments = process_ground_truths(name_list, gt_list, out_list, mom_list, config)

# %%===========================================================================
# Compute Blur measure
name_list = [os.path.join(config["OUTPUT_PATH"], "preprocessed_" + str(x) + ".tif") for x in range(config["N_IMAGES"])]
# name_list = [os.path.join(config["OUTPUT_PATH"], "split_" + str(x) + ".tif") for x in range(config["N_IMAGES"])]
images = read_list_of_images(name_list)

rectangle = config['CROP_RECTANGLE']
sizes = (rectangle[2]/2 * np.linspace(0, 1.5, 21)).astype('int')
# size = int(percentages[0])

def plotyy(x, y1, y2, parameters=dict()):
    return 0

distances = []
correlations = []
for size in sizes:
    blurs = []
    mags  = []
    rectangle = config['CROP_RECTANGLE']
    for i, image in enumerate(images):
            
        blur, _, mag = detect_blur_fft(image, size=size, verbose=True)
        mags.append(mag)
        blurs.append(blur)
    
    series = pd.Series(blurs)
    
    series.to_csv(os.path.join(config["OUTPUT_PATH"], config["TITLE"] + f"_blurscore_size_{size:04}.csv"), header=None, index=None)
    
    ind = range(config["N_IMAGES"])
    fig, ax = plt.subplots(figsize=(16,8))
    
    a = areas.copy()
    s = series.tolist()
    series_norm = (s - np.min(s)) / (np.max(s) - np.min(s))
    areas_norm  = (a - np.min(a)) / (np.max(a) - np.min(a))
    distance = np.linalg.norm(series_norm - areas_norm, ord=2)
    distances.append(distance)
    
    r, p = compute_correlation(s, a)
    correlations.append((r, p))

    # color = 'tab:red'
    # ax.plot(ind, blurs, color=color, marker='s', linestyle='dotted')
    # ax.set_xticks(ind)
    # ax.set_xticklabels(ind)
    # ax.set_xlabel("Image index")
    # ax.set_ylabel("Blur score", color=color)
    # ax.tick_params(axis='y', labelcolor=color)
    
    # ax2 = ax.twinx()
    # color = 'tab:blue'
    # ax2.plot(ind, areas, color=color, marker='o', linestyle='solid')
    # ax2.set_xticks(ind)
    # ax2.set_xticklabels(ind, color=color)
    # ax2.set_xlabel("Image index")
    # ax2.set_ylabel("Ground Truth Area (Pixels)", color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    
    # plt.title(config["TITLE"])
    
    

    
    ind = range(config["N_IMAGES"])
    fig, ax = plt.subplots()
    
    color = 'tab:red'
    ax.plot(ind, series_norm, color=color, marker='s', linestyle='dotted', label="Blur score")
    ax.set_xticks(ind)
    ax.set_xticklabels(ind)
    ax.set_xlabel("Image index")
    
    color = 'tab:blue'
    ax.plot(ind, areas_norm, color=color, marker='o', linestyle='solid', label="Ground Truth Area (Pixels)")
    ax.set_xticks(ind)
    ax.set_xlabel("Image index")
    
    plt.title(config["TITLE"])
    plt.savefig(os.path.join(config["OUTPUT_PATH"], config["TITLE"] + f"_size_{size:04}.png"), dpi=200)
    plt.legend()
    plt.savefig(os.path.join(config["OUTPUT_PATH"], config["TITLE"] + f"_size_{size:04}.png"), dpi=200)

distances = pd.Series(distances)
distances.to_csv(os.path.join(config["OUTPUT_PATH"], config["TITLE"] + f"_distances.csv"), header=None, index=None)
plt.figure()
plt.plot(sizes, distances, 'ro')
plt.title(config["TITLE"])
plt.savefig(os.path.join(config["OUTPUT_PATH"], config["TITLE"] + f"_distances.png"))

corrs = pd.DataFrame(correlations)
corrs.to_csv(os.path.join(config["OUTPUT_PATH"], config["TITLE"] + f"_correlations.csv"), header=None, index=None)



#%%

# series_norm = (s - np.min(s)) / (np.max(s) - np.min(s))
# areas_norm  = (a - np.min(a)) / (np.max(a) - np.min(a))
# distance = np.linalg.norm(series_norm - areas_norm, ord=2)

# ind = range(config["N_IMAGES"])
# fig, ax = plt.subplots()

# color = 'tab:red'
# ax.plot(ind, series_norm, color=color, marker='s', linestyle='dotted', label="Blur score")
# ax.set_xticks(ind)
# ax.set_xticklabels(ind)
# ax.set_xlabel("Image index")

# color = 'tab:blue'
# ax.plot(ind, areas_norm, color=color, marker='o', linestyle='solid', label="Ground Truth Area (Pixels)")
# ax.set_xticks(ind)
# # ax.set_xticklabels(ind, color=color)
# ax.set_xlabel("Image index")

# plt.title(config["TITLE"])
# plt.savefig(os.path.join(config["OUTPUT_PATH"], config["TITLE"] + f"_size_{size:04}.png"), dpi=200)
# plt.legend()
# plt.show()



# %%===========================================================================
# Binarize all pre-processed images
# preprocessed_filenames = [os.path.join(config["OUTPUT_PATH"], "preprocessed_" + str(x) + ".tif") for x in range(config["N_IMAGES"])]
# images = read_list_of_images(preprocessed_filenames)

# binarized_filenames = [os.path.join(config["OUTPUT_PATH"], "binarized_" + str(x) + ".tif") for x in range(config["N_IMAGES"])]

# for image, output_filename in zip(images, binarized_filenames):
    
#     _, im_b = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
#     cv2.imwrite(output_filename, im_b)
#     print(output_filename)
    

# %%===========================================================================
# Compute snakes onto image.
# """preprocessed_filenames = [os.path.join(config["OUTPUT_PATH"], "preprocessed_" + str(x) + ".tif") for x in range(config["N_IMAGES"])]"""
# """preprocessed_images = read_list_of_images(preprocessed_filenames)"""
# binarized_filenames = [os.path.join(config["OUTPUT_PATH"], "binarized_" + str(x) + ".tif") for x in range(config["N_IMAGES"])]
# binarized_images = read_list_of_images(binarized_filenames)

# for k in range(config["N_IMAGES"]):
    
#     print(f"Processing {k}th image file.")
    
#     # Compute snakes
#     """image = 1.0 - normalize_image(preprocessed_images[k])"""
#     image = 1.0 - normalize_image(binarized_images[k])
#     evolution = compute_snakes(image, threshold=config["SNAKES_THRESH"], n_iter=config["SNAKES_N_ITER"])
#     evolution = np.array(evolution)

#     # Save binary image containing contour corresponding to last SNAKES iteration
#     output_filename = os.path.join(config["OUTPUT_PATH"], "snakes_" + str(k) + ".tif")
#     cv2.imwrite(output_filename, (255*evolution[-1]).astype(np.uint8))
    
#     # Plot intermediate SNAKES iterations
#     fig = plt.figure(k, figsize=(10,5))
#     plt.imshow(image, cmap="gray")
#     iterations = np.linspace(0, config["SNAKES_N_ITER"], config["SNAKES_N_CONTOURS"]).astype(int)
#     colors = ['k', 'y', 'g', 'c', 'm', 'r']
#     for ind, i in enumerate(iterations):
#         ax = fig.axes[0]
#         contour = ax.contour(evolution[i], [.5], colors=colors[ind])
#         contour.collections[0].set_label(f"It. {i}")
#     # plt.imshow(evolution[i], cmap='gray')
#     title = "Morphological GAC evolution"
#     ax.contour(evolution[-1], [0.5], colors='r')
#     ax.legend(bbox_to_anchor=(1.04,0.7), loc="upper left")
#     fig.tight_layout()
#     plt.gca().set_aspect('equal')
#     plt.gca().set_axis_off()
#     # plt.show()
#     # Save figures
#     output_figname = os.path.join(config["OUTPUT_PATH"], "snakes_fig_" + str(k))
#     plt.savefig(output_figname)

