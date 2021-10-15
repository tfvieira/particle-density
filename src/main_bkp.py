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

from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

from utils  import *
from filetools import *

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

# %%===========================================================================
# Split one TIF image into many images, each corresponding to one TIF layer
split_images(config['INPUT_FILENAME'], 
            os.path.join(config['OUTPUT_PATH'], 'split'))

#%%
name_list = [os.path.join(config['OUTPUT_PATH'], 'split', 'split_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
images = read_list_of_images(name_list)


global n_clicks, mouseX, mouseY, image_1, image_2

ind = 0
while ind < config['N_IMAGES']:

    image    = images[ind].copy()
    image_1  = image.copy()
    image_2  = image.copy()
    n_clicks = int(0)

    posList = []
    def draw_circle(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            print('x = %d, y = %d'%(x, y))
            posList.append((x, y))
            cv2.drawMarker(image_1, (x,y), (255,0,0), cv2.MARKER_CROSS, 10, 1)
            global n_clicks
            n_clicks = n_clicks + 1
            mouseX, mouseY = x, y

    cv2.namedWindow('image_1', cv2.WINDOW_KEEPRATIO)
    # cv2.setWindowProperty('image_1', cv2.WINDOW_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.namedWindow('image_2', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('image_1', draw_circle)

    # histogram = draw_histogram(image)

    while(True):

        cv2.imshow('image_1'  , image_1)
        # cv2.imshow('image_2'  , image_2)
        # cv2.imshow('histogram', histogram)

        key = 0xFF & cv2.waitKey(20)

        if key == ord('n'):
            ind = ind + 1
            break

        elif key == ord('q'):
            
            cv2.destroyAllWindows()
            break

        elif n_clicks == 5:

            posNp = np.array(posList)
            ellipse = fit_ellipses([posNp])
            image_2 = draw_ellipses(image_1, ellipse)

            # init_level_set = np.zeros(image.shape, dtype=np.int8)
            # init_level_set = draw_ellipses(image_2, ellipse)

            filename = os.path.join(config['OUTPUT_PATH'], 'ellipse', 'ellipse_' + str(ind) + '.json')
            write_json(ellipse[0], filename)
                    

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


# %%===========================================================================
name_list  = [os.path.join(config['OUTPUT_PATH'], 'ellipse', 'ellipse_' + str(x) + '.json') for x in range(config['N_IMAGES'])]
image_list = [os.path.join(config['OUTPUT_PATH'], 'split', 'split_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
filenames  = [os.path.join(config['OUTPUT_PATH'], 'crop_centered', 'crop_centered_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]

for i in range(config['N_IMAGES']):

    json_file       = name_list[i]
    image_file      = image_list[i]
    output_filename = filenames[i]

    ellipse = read_json(json_file)
    x, y = ellipse[0][:]
    w, h = config['CROP_RECTANGLE'][2:]

    x0 = int(x - w/2)
    xf = int(x + w/2)
    y0 = int(y - w/2)
    yf = int(y + w/2)

    image         = cv2.imread(image_file)
    cropped_image = image[y0:yf, x0:xf]

    # Show results
    cv2.namedWindow('image'        , cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('cropped_image', cv2.WINDOW_KEEPRATIO)

    cv2.imshow('image', image)
    cv2.imshow('cropped_image', cropped_image)

    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    cv2.imwrite(output_filename, cropped_image)

#%%
filenames  = [os.path.join(config['OUTPUT_PATH'], 'crop_centered', 'crop_centered_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
images = read_list_of_images(filenames)
show_list_of_images(images)














# %%===========================================================================
# Crop the images to contain only the particles
crop_images(config['INPUT_FILENAME'], 
            os.path.join(config['OUTPUT_PATH'], 'crop'), 
            rectangle=config['CROP_RECTANGLE'])

# %%===========================================================================
# Blur the images
name_list = [os.path.join(config['OUTPUT_PATH'], 'crop', 'crop_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
images = read_list_of_images(name_list)
images = blur_list_of_images(images)
filenames = [os.path.join(config['OUTPUT_PATH'], 'blur', 'blur_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
write_list_of_images(filenames, images)


# %%===========================================================================
# Equalize the cropped images to reduce noise
name_list = [os.path.join(config['OUTPUT_PATH'], 'crop', 'crop_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
images = read_list_of_images(name_list)
images = equalize_list_of_images(images)
filenames = [os.path.join(config['OUTPUT_PATH'], 'equalize', 'equalize_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
write_list_of_images(filenames, images)





# %%===========================================================================
# Find the Laplacian
name_list = [os.path.join(config['OUTPUT_PATH'], 'blur', 'blur_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
images = read_list_of_images(name_list)
images = laplacian_list_of_images(images)
filenames = [os.path.join(config['OUTPUT_PATH'], 'laplacian', 'laplacian_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
write_list_of_images(filenames, images)


# %%===========================================================================
# # Show list of pre-processed images
filenames = [os.path.join(config['OUTPUT_PATH'], 'crop', 'crop_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
images = read_list_of_images(filenames)
show_list_of_images(images)









# %%===========================================================================
# Pre-process all images
name_list = [os.path.join(config['OUTPUT_PATH'], 'crop', 'crop_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
images = read_list_of_images(name_list)
preprocessed_images = preprocess_list_of_images(images)
preprocessed_filenames = [os.path.join(config['OUTPUT_PATH'], 'preprocessed', 'preprocessed_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
write_list_of_images(preprocessed_filenames, preprocessed_images)


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

#%%
areas = np.zeros(config['N_IMAGES'])

#%%
filenames = [os.path.join(config['OUTPUT_PATH'], 'crop', 'crop_' + str(x) + '.tif') for x in range(config['N_IMAGES'])]
images = read_list_of_images(filenames)

ind = 1 # 4, 5, 8, 9, 10, 11-19
image = images[ind]
img = rgb2gray(image)

s = np.linspace(0, 2*np.pi, 400)
r = 150 + 150*np.sin(s)
c = 200 + 150*np.cos(s)
init = np.array([r, c]).T

if ind in range(4) or ind in range(6,10):
    snake_pars = {'alpha': 0.015, 'beta': 1, 'gamma': 0.001, 'w_edge': 1}
elif ind == 4:
    snake_pars = {'alpha': 0.020, 'beta': 10, 'gamma': 0.0019, 'w_edge': 1}
elif ind == 10:
    snake_pars = {'alpha': 0.02, 'beta': 25, 'gamma': 0.01, 'w_edge': 1}
else:
    snake_pars = {'alpha': 0.02, 'beta': 25, 'gamma': 0.1, 'w_edge': 1}

snake = active_contour(
    gaussian(img, 3, preserve_range=False),
    init, alpha=snake_pars['alpha'],
    beta=snake_pars['beta'],
    gamma=snake_pars['gamma'],
    w_edge=snake_pars['w_edge']
)

M = cv2.moments(snake)
areas[ind] = int(M['m00'])

fig, ax = plt.subplots(1,2, figsize=(12, 6))
ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].plot(init[:, 1], init[:, 0], '--r', lw=3)
ax[0].plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax[0].axis([0, img.shape[1], img.shape[0], 0])
ax[0].grid(False)
ax[1].stem(range(config['N_IMAGES']), areas)
plt.show()

# %%
