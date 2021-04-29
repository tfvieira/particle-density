# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 10:15:56 2021

@author: Vieira
"""

#%%
# import cv2
import numpy as np
from matplotlib import pyplot as plt

#%%
def detect_blur_fft(image, size=0, thresh=10, plot_results=False, verbose=False):
    """
    Assign 

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    size : TYPE, optional
        DESCRIPTION. The default is 0.
    thresh : TYPE, optional
        DESCRIPTION. The default is 10.
    plot_results : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    mean : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """

    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # check to see if we are visualizing our output
    if plot_results:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(1 + np.abs(fftShift))

        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        # show our plots
        plt.show()

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(1 + recon))
    mean = np.mean(magnitude)
    

    # Print the mean if verbose is TRUE
    if verbose is True:
        print(f"FFT mean = {mean}")

    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh, magnitude)



# blurs = []
# for i in range(15,5000, 5):
    
#     img = np.random.random((150,250))
#     blur = detect_blur_fft(img, size=0)[0]
#     blurs.append(blur)
#     print(f"i = {i},\tBlur = {blur}")

# #%%
# import pandas as pd
# series = pd.Series(blurs)
# series.plot.box()















