# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:49:53 2021

@author: Vieira
"""

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)


#%%
def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store

#%%
# _, output = cv2.imreadmulti("../results/pre_processed.tif", [], cv2.IMREAD_GRAYSCALE)


def compute_snakes(image, threshold=0.83, n_iter=400, verbose=True):
    
    # Initial level set
    init_level_set = np.zeros(image.shape, dtype=np.int8)
    init_level_set[10:-10, 10:-10] = 1
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_geodesic_active_contour(image, n_iter, init_level_set,
                                               smoothing=2, balloon=-1,
                                               threshold=threshold,
                                               iter_callback=callback)
    return evolution


# #%%

# ind = 0

# print(f"Processing index = {ind}")

# image = 1.0 - cv2.normalize(output[ind].astype(np.float), None, 1, 0, cv2.NORM_MINMAX)
# gimage = image.copy()#inverse_gaussian_gradient(image)

# # Initial level set
# init_ls = np.zeros(image.shape, dtype=np.int8)
# init_ls[10:-10, 10:-10] = 1
# # List with intermediate results for plotting the evolution
# evolution = []
# callback = store_evolution_in(evolution)
# ls = morphological_geodesic_active_contour(gimage, 400, init_ls,
#                                            smoothing=2, balloon=-1,
#                                            threshold=0.9,
#                                            iter_callback=callback)

# fig, axes = plt.subplots(1, 2, figsize=(8, 8))
# ax = axes.flatten()

# ax[0].imshow(gimage, cmap="gray")
# ax[0].set_axis_off()
# ax[0].contour(ls, [0.5], colors='r')
# ax[0].set_title("Morphological GAC segmentation", fontsize=12)

# ax[1].imshow(ls, cmap="gray")
# ax[1].set_axis_off()

# contour = ax[1].contour(evolution[0], [0.5], colors='g')
# contour.collections[0].set_label("Iteration 0")
# contour = ax[1].contour(evolution[100], [0.5], colors='y')
# contour.collections[0].set_label("Iteration 100")
# contour = ax[1].contour(evolution[-1], [0.5], colors='r')
# contour.collections[0].set_label("Iteration 500")
# ax[1].legend(loc="upper right")
# title = "Morphological GAC evolution"
# ax[1].set_title(title, fontsize=12)

# fig.tight_layout()
# # plt.savefig(f"../results/{ind}")
# # cv2.imwrite(f"../results/snakes_2_ind_{ind}.png", (255*ls).astype(np.uint8))

# plt.show()
