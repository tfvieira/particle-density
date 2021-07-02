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
import shutil

from utils  import * #read_list_of_images
from snakes import *
from detect_blur_fft import detect_blur_fft

plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Define IO parameters
CONFIG_PATH = 'config'

EXPERIMENTS = [
    '10-microns particles-60X',
    'Calibration2_Single Cell',
    'Calibration1_Single Cell',
    'Four-mixing particles together',
    'Several 10-micron-particles together',
    'Calibration 10-microns',
    '30 microns-beads-60X-measuring 2',
    'Calibration-1-4 Cells',
    'Isolada 3--3',
    'Isolada 3--2',
    'Isolada-2-10 um',
    '3 particles_10 um',
]

#%%
# for EXPERIMENT in EXPERIMENTS:
EXPERIMENT = EXPERIMENTS[0]

# Read configuration parameters from JSON file
CONFIG_FILENAME = os.path.join(CONFIG_PATH, EXPERIMENT + '.json')
with open(CONFIG_FILENAME, 'r') as fp:
    config = json.load(fp)

# %% Split one TIF image into many images, each corresponding to one TIF layer
split_images(config['INPUT_FILENAME'], 
            os.path.join(config['OUTPUT_PATH'], 'split'))

# %%===========================================================================
# Crop the images to contain only the particles
crop_images(config['INPUT_FILENAME'], 
            os.path.join(config['OUTPUT_PATH'], 'crop'), 
            rectangle=config['CROP_RECTANGLE'])

# %%===========================================================================
# Pre-process all images
name_list = [os.path.join(config['OUTPUT_PATH'], 'crop', 'crop_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
images = read_list_of_images(name_list)
preprocessed_images = preprocess_list_of_images(images)
preprocessed_filenames = [os.path.join(config['OUTPUT_PATH'], 'preprocessed', 'preprocessed_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
write_list_of_images(preprocessed_filenames, preprocessed_images)

# %===========================================================================
# # Show list of pre-processed images
preprocessed_filenames = [os.path.join(config['OUTPUT_PATH'], 'crop', 'crop_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
# preprocessed_filenames = [os.path.join(config['OUTPUT_PATH'], 'preprocessed', 'preprocessed_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
preprocessed_images = read_list_of_images(preprocessed_filenames)
show_list_of_images(preprocessed_images)

# %%===========================================================================
# Process Ground truths
if config['GROUND_TRUTHS_AVAILABLE'] == True:
    name_list = [os.path.join(config['OUTPUT_PATH'], 'split', 'split_' + str(x) + '.tif')       for x in range(config['N_IMAGES'])]
    gt_list   = [os.path.join(config['OUTPUT_PATH'], 'gt',    'gt_' + str(x) + '.tif')          for x in range(config['N_IMAGES'])]
    out_list  = [os.path.join(config['OUTPUT_PATH'], 'split_gt', 'split_gt_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
    mom_list  = [os.path.join(config['OUTPUT_PATH'], 'moments', 'moments_' + str(x) + '.json')  for x in range(config['N_IMAGES'])]
    areas, moments = process_ground_truths(name_list, gt_list, out_list, mom_list, config)

    write_json(areas, os.path.join(config['OUTPUT_PATH'], 'gt', 'areas.json'))
    write_json(moments, os.path.join(config['OUTPUT_PATH'], 'gt', 'moments.json'))


#%% Compute Blur measure
# name_list = [os.path.join(config['OUTPUT_PATH'], 'preprocessed', 'preprocessed_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
name_list = [os.path.join(config['OUTPUT_PATH'], 'crop', 'crop_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
images = read_list_of_images(name_list)

rectangle = config['CROP_RECTANGLE']
sizes = (rectangle[2]/2 * np.linspace(0.1, 1.5, 21)).astype('int')

if config['GROUND_TRUTHS_AVAILABLE'] == True:
    areas = read_json(os.path.join(config['OUTPUT_PATH'], 'gt', 'areas.json'))
    distances = []
    correlations = []

for size in sizes:

    blurs = []
    mags  = []
    rectangle = config['CROP_RECTANGLE']
    
    for i, image in enumerate(images):
            
        # blur, _, mag = detect_blur_fft(image, size=size, verbose=True)
        # mags.append(mag)
        image = cv2.medianBlur(image, 27)
        blur = compute_blur(image, gaussian_sigma = size)
        blurs.append(blur[0])
    
    series = pd.Series(blurs)
    
    series.to_csv(os.path.join(config['OUTPUT_PATH'], 'blurscores', f'blurscore_size_{size:04}.csv'), header=None, index=None)
    
    s = series.tolist()
    series_norm = (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-22)

    if config['GROUND_TRUTHS_AVAILABLE'] == True:
        a = areas.copy()
        areas_norm  = (a - np.min(a)) / (np.max(a) - np.min(a))
        distance = np.linalg.norm(series_norm - areas_norm, ord=2)
        distances.append(distance)
        r, p = compute_correlation(s, a)
        correlations.append((r, p))

    ind = range(config['N_IMAGES'])

    fig, ax = plt.subplots()
    
    color = 'tab:red'
    ax.plot(ind, series_norm, color=color, marker='s', linestyle='dotted', label='Blur score')
    ax.set_xticks(ind)
    ax.set_xticklabels(ind)
    ax.set_xlabel('Image index')
    
    if config['GROUND_TRUTHS_AVAILABLE'] == True:
        color = 'tab:blue'
        ax.plot(ind, areas_norm, color=color, marker='o', linestyle='solid', label='Ground Truth Area (Pixels)')
        ax.set_xticks(ind)
        ax.set_xlabel('Image index')
    
    plt.title(config['TITLE'])
    plt.legend()
    plt.savefig(os.path.join(config['OUTPUT_PATH'], 'blurscores', f'blurscore_size_{size:04}.png'), dpi=200)

# # distances = pd.Series(distances)
# # distances.to_csv(os.path.join(config['OUTPUT_PATH'], config['TITLE'] + f'_distances.csv'), header=None, index=None)
# # plt.figure()
# # plt.plot(sizes, distances, 'ro')
# # plt.title(config['TITLE'])
# # plt.savefig(os.path.join(config['OUTPUT_PATH'], config['TITLE'] + f'_distances.png'))

# # corrs = pd.DataFrame(correlations)
# # corrs.to_csv(os.path.join(config['OUTPUT_PATH'], config['TITLE'] + f'_correlations.csv'), header=None, index=None)

#%% Concatenate all blur scores
rectangle = config['CROP_RECTANGLE']
sizes = (rectangle[2]/2 * np.linspace(0.1, 1.5, 21)).astype('int')
# sizes = [config['BEST_BLURSCORE']]
csv_filenames = [os.path.join(config['OUTPUT_PATH'], 'blurscores', f'blurscore_size_{size:04}.csv') for size in sizes]
df = read_csv_files(csv_filenames)
df.columns = sizes
df_norm = normalize_df_min_max(df)

# Save all blurscores
df.to_csv(os.path.join(config['OUTPUT_PATH'], 'blurscores', f'blurscores.csv'), header=None, index=None)
df_norm.to_csv(os.path.join(config['OUTPUT_PATH'], 'blurscores', f'blurscores_norm.csv'), header=None, index=None)

df.plot()
plt.savefig(os.path.join(config['OUTPUT_PATH'], 'blurscores', f'blurscores.png'))
df.transpose().plot.box()
plt.savefig(os.path.join(config['OUTPUT_PATH'], 'blurscores', f'blurscores_boxplot.png'))

df_norm.plot()
plt.savefig(os.path.join(config['OUTPUT_PATH'], 'blurscores', f'blurscores_norm.png'))
df_norm.transpose().plot.box()
plt.savefig(os.path.join(config['OUTPUT_PATH'], 'blurscores', f'blurscores_norm_boxplot.png'))


#%% Calibrate
df = pd.read_csv(os.path.join(config['OUTPUT_PATH'], 'blurscores', f'blurscores.csv'), header=None)
df_norm = pd.read_csv(os.path.join(config['OUTPUT_PATH'], 'blurscores', f'blurscores_norm.csv'), header=None)
df = normalize_df_min_max(df)
m = df.transpose().mean()

if config['GROUND_TRUTHS_AVAILABLE'] == True:
    areas = read_json(os.path.join(config['OUTPUT_PATH'], 'gt', 'areas.json'))
    areas  = (areas - np.min(areas)) / (np.max(areas) - np.min(areas))

N = config['N_IMAGES']

x = np.linspace(0,N,N)
x_curve_fit = np.linspace(0, N, 200)

if config['GROUND_TRUTHS_AVAILABLE'] == True:
    pars_areas, cov_areas = fit_single_exponential(x, areas, maxfev=1000)
    curve_fit_areas = exponential(x_curve_fit, *pars_areas)

pars_blurs, cov_blurs = fit_single_exponential(x, m, maxfev=1000)
curve_fit_blurs = exponential(x_curve_fit, *pars_blurs)

if config['GROUND_TRUTHS_AVAILABLE'] == True:
    d = {
        'blur': {
            'pars' : {'a': pars_blurs[0],  'b' : pars_blurs[1]},
            'cov'  : {'a': cov_blurs[0,0], 'b' : cov_blurs[1,1]},
            },
        'gt'  : {
            'pars' : {'a' : pars_areas[0],  'b' : pars_areas[1]},
            'cov'  : {'a' : cov_areas[0,0], 'b' : cov_areas[1,1]}
        }
    }
else:
    d = {
        'blur': {
            'pars' : {'a': pars_blurs[0],  'b' : pars_blurs[1]},
            'cov'  : {'a': cov_blurs[0,0], 'b' : cov_blurs[1,1]},
            },
        'gt'  : {
            'pars' : {'a' : None, 'b' : None},
            'cov'  : {'a' : None, 'b' : None}
        }
    }

write_json(d, os.path.join(config['OUTPUT_PATH'], 'calibration', f'calibration.json'))

plt.figure(figsize=(6,4))
if config['GROUND_TRUTHS_AVAILABLE'] == True:
    plt.plot(x, areas, 'bd', label='Area')
if config['GROUND_TRUTHS_AVAILABLE'] == True:
    plt.plot(x_curve_fit, curve_fit_areas, '-b', label='Exponential fit (Areas)')
plt.plot(x, m, 'ro', label='Blur score')
plt.plot(x_curve_fit, curve_fit_blurs, '-r', label='Exponential fit (Blurs)')
plt.xlabel('Image index')
plt.ylabel('Normalized units')
plt.legend()
plt.title(config['TITLE'])
plt.savefig(os.path.join(config['OUTPUT_PATH'], 'calibration', 'calibration.png'))


plt.savefig(os.path.join('calibrations', 'calibration_' + config['TITLE'] + '.png'))


#%%

# heights = np.linspace(125, 91, 18) # Calibration Heights
# pars, cov = curve_fit(f=double_exponential, xdata=m.flatten(), ydata=heights, p0=np.random.random((1,5)), bounds=(-np.inf, np.inf))

# x = np.linspace(0,1,100)
# y = double_exponential(x, pars[0], pars[1], pars[2], pars[3], pars[4])

# #%%
# fig = plt.figure(figsize=(6,8))
# ax = fig.add_subplot()
# ax.plot(m, heights, 'k.')
# ax.plot(x, y, '-r', label = f'{pars[0]:.2}exp(-x/{pars[1]:.2}) + {pars[2]:.2}exp(-x/{pars[3]:.2}) + {pars[4]:.2}')
# plt.legend()
# plt.ylabel('Height ($\mu m$)')
# plt.xlabel('Normalized Blur Score')
# plt.show()


# #%%
# linestyles = ['-', '--', '-.', ':']
# markers    = ['s', 'o', 'D', '*']
# colors     = ['b', 'g', 'r', 'm']
# heights = np.linspace(125, 91, 18) # Calibration Heights

# i = 0
# fig = plt.figure(figsize=(6,8))
# ax = fig.add_subplot()

# parameters  = []
# covariances = []

# x = np.linspace(-0,1,100)



# for key, arr in blurscores.items():

#     print(f'{len(arr)}, {key}')

#     d = arr[:18]
#     d = (d - d[0]) / (d.max() - d.min())

#     pars, cov = curve_fit(f=exponential, xdata=d.flatten(), ydata=heights, p0=[0, 0, 0], bounds=(-np.inf, np.inf))

#     parameters.append(pars)
#     covariances.append(cov)

#     y = exponential(x, pars[0], pars[1], pars[2])

#     ax.plot(
#         d, heights, 
#         linestyle = '',
#         # linestyle  = linestyles[i],
#         marker     = markers[i],
#         color      = colors[i],
#         label      = key)

#     ax.plot(
#         x, y, linestyle = linestyles[i], color=colors[i],
#         label = f'{pars[0]:.2}exp({pars[1]:.2}x) + {pars[2]:.2}'
#     )

#     i += 1

# plt.legend(loc='upper right')
# plt.ylabel('Height ($\mu m$)')
# plt.xlabel('Normalized Blur Score')
# plt.show()


#%%
# for EXPERIMENT in EXPERIMENTS:

#     CONFIG_FILENAME = os.path.join(CONFIG_PATH, EXPERIMENT + '.json')
#     with open(CONFIG_FILENAME, 'r') as fp:
#         config = json.load(fp)

#     src = os.path.join(config['OUTPUT_PATH'], 'calibration', 'calibration.png')
#     dst = 'calibration_' + config['TITLE'] + '.png'
#     shutil.copyfile(src, dst)
#     print(dst)




























# %%===========================================================================
# Binarize all pre-processed images
# preprocessed_filenames = [os.path.join(config['OUTPUT_PATH'], 'preprocessed_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
# images = read_list_of_images(preprocessed_filenames)

# binarized_filenames = [os.path.join(config['OUTPUT_PATH'], 'binarized_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]

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

