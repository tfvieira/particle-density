# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:01:52 2021

@author: Vieira
"""

#%% Import packages
import os
import cv2
import json
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from detect_blur_fft import detect_blur_fft
from sklearn import preprocessing
from scipy.optimize import curve_fit
from detect_blur_fft import detect_blur_fft

plt.rcParams.update({'figure.max_open_warning': 0})

#%%
def read_json(json_filename):

    with open(json_filename, 'r') as fp:
        x = json.load(fp)
    
    return x

def write_json (x, json_filename, sort_keys=True, indent=4):
    
    with open(json_filename, 'w') as fp:
        json.dump(x, fp, sort_keys=sort_keys, indent=indent)
    
    return None


def create_2D_gaussian(
    shape = (100, 100), 
    mx = 50, 
    my = 50, 
    sx = 10, 
    sy = 10,
    theta = 0):
    """
    Create an image with shape = (rows x cols) with a 2D Gaussian with
    mx, my means in the x and y directions and standard deviations
    sx, sy respectively. The Gaussian can also be rotate of theta
    radians in clockwise direction.

    Example usage:
    g = create_2D_gaussian(
        shape = (500, 1000), 
        mx = 5000, 
        my = 250, 
        sx = 60, 
        sy = 20,
        theta = -30
        )
    """
    
    xx0, yy0 = np.meshgrid(range(shape[1]), range(shape[0]))
    xx0 -= int(mx)
    yy0 -= int(my)
    theta = np.deg2rad(theta)
    xx = xx0 * np.cos(theta) - yy0 * np.sin(theta)
    yy = xx0 * np.sin(theta) + yy0 * np.cos(theta)
    try:
        img = np.exp( - ((xx**2)/(2*sx**2) + 
                         (yy**2)/(2*sy**2)) )
    except ZeroDivisionError:
        img = np.zeros((shape[0], shape[1]), dtype='float64')

    return cv2.normalize(img.astype('float'), None, 1, 0, cv2.NORM_MINMAX)




def compute_histogram_1C(src):
    # Compute the histograms:
    b_hist = cv2.calcHist([src], [0], None, [256], [0, 256], True, False)

    # Draw the histograms for B, G and R
    hist_w = 512
    hist_h = 400
    bin_w = np.round(hist_w / 256)

    histImage = np.ones((hist_h, hist_w), np.uint8)

    # Normalize the result to [ 0, histImage.rows ]
    cv2.normalize(b_hist, b_hist, 0, histImage.shape[0], cv2.NORM_MINMAX)

    # Draw for each channel
    for i in range(1, 256):
        cv2.line(histImage, (int(bin_w * (i - 1)), int(hist_h - np.round(b_hist[i - 1]))),
                 (int(bin_w * i), int(hist_h - np.round(b_hist[i]))), 255, 2, cv2.LINE_8, 0)
    return histImage



def split_images(input_filename, output_path, output_name="split", verbose=True):

    _, lst = cv2.imreadmulti(input_filename, [], cv2.IMREAD_GRAYSCALE)

    for i, img in enumerate(lst):
        output_filename = os.path.join(output_path, output_name + "_" + str(i) + ".tif")
        if verbose == True:
            print(f"Splitting layer {i} of image file {output_filename}.")
        cv2.imwrite(output_filename, img)
        
    return None

def crop_images(input_filename, output_path, rectangle, output_name="crop", verbose=True):

    _, lst = cv2.imreadmulti(input_filename, [], cv2.IMREAD_GRAYSCALE)
    
    x, y, w, h = rectangle
    for i, img in enumerate(lst):
        output_filename = os.path.join(output_path, output_name + "_" + str(i) + ".tif")
        output_image = img[y:y+h, x:x+w]
        if verbose == True:
            print(f"Cropping layer {i} of image file {output_filename}.")
        cv2.imwrite(output_filename, output_image)
    
    return None

def normalize_image(image):
    """
    Normalize image to [0,1] range.
    """
    return cv2.normalize(image.astype('float64'), None, 1, 0, cv2.NORM_MINMAX)




def do_nothing(x):
    pass

def read_list_of_images(filenames):
    """
    Read a list of image files specified in list FILENAMES.
    """
    
    images = []
    
    for filename in filenames:
        print(f"Reading image {filename}")
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        images.append(image)
    
    return images

def find_external_contours(binary_image):

    _, contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def exponential( x, a, b ):
    """
    Define a single exponential function.
    """
    return a * ( np.exp( b * x ) )

def double_exponential(x, a1, t1, a2, t2, y0):
    """
    Define a double exponential function.
    """
    return a1 * np.exp( - x/t1 ) + a2 * np.exp( - x/t2 ) + y0


def power_law(x, a, b):
    """
    Define a single power law function.
    """
    return a * np.power(x, b)



def fit_single_exponential (x, y, maxfev = 1e3):
    """
    Fit a single exponential to a set of data points defined by _x_ and _y_.
    """
    pars, cov = curve_fit(
        f      = exponential,
        xdata  = x,
        ydata  = y,
        p0     = [0, 0], 
        bounds = (-np.inf, np.inf), 
        maxfev = maxfev,
        )

    a, b = pars
    s1 = np.sqrt(cov[0,0])
    s2 = np.sqrt(cov[0,0])
    print(f'a = {a:.4f} +- {s1:.4f}')
    print(f'b = {b:.4f} +- {s2:.4f}')

    return pars, cov

def plot_data_and_single_exponential(x_data, y_data, x_curve_fit, y_curve_fit):
    """
    Plot a scatter data _x_ and _y_ and the correponding single exponential fit
    """

    fig, ax = plt.subplots()
    ax.plot(
        x_data, y_data, 
        marker='.', markersize=10, color='#00b3b3', 
        label='Data',
        linestyle = '-', linewidth = .5)

    ax.plot(x_curve_fit, y_curve_fit, '-b', label='$a\cdot e^{b x}$')

    # ax.set_yscale('log')
    # ax.set_ylim(1e-2, 1)
    # plt.grid(b = True, which = 'minor')

    plt.ylabel('Blur score')
    plt.xlabel('Image index')
    plt.legend()
    plt.show()

    return None


def tuple2ellipse(e):

    cx = int(e[0][0])
    cy = int(e[0][1])
    w  = int(e[1][0]/2)
    h  = int(e[1][1]/2)
    a  = int(e[2])
    
    return ((cx, cy), (w, h), a)

def fit_ellipses(contours):

    ellipses = []

    for contour in contours:
        
        if len(contour) >= 5:               # Fitting algorithm needs at least 5 points.
            e0 = cv2.fitEllipse(contour)    # Fit ellipse
            ellipses.append(tuple2ellipse(e0))

    return ellipses



def draw_ellipses(img, ellipses, color = 255):

    for e in ellipses:

        cv2.ellipse(img, e[0], e[1], e[2], 0, 360, color, 2)
    
    return img


def find_contours_and_draw_ellipses(binary_image):

    contours = find_external_contours(binary_image)
    
    return draw_ellipses(binary_image, fit_ellipses(contours))



def show_list_of_images(images, ind = 0, med_blur_size=27, ksize=31, gauss_size = 20, thresh=0, circles=None):
    
    cv2.namedWindow('sliders'   , cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('gray'      , cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('diff'      , cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('mag_grad'  , cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('thresh'    , cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('fft_shift' , cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('fft_hist'  , cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('ellipses'  , cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    
    gauss_size_2 = 5
    cv2.createTrackbar('ind'           , 'sliders', ind             , len(images)-1, do_nothing)
    cv2.createTrackbar('med_blur_size' , 'sliders', med_blur_size   , 300,           do_nothing)
    cv2.createTrackbar('ksize'         , 'sliders', ksize           , 31 ,           do_nothing)
    cv2.createTrackbar('gauss_size'    , 'sliders', gauss_size      , 500,           do_nothing)
    cv2.createTrackbar('gauss_size_2'  , 'sliders', gauss_size_2    , 500,           do_nothing)
    # cv2.createTrackbar('thresh'        , 'sliders', thresh          , 255,           do_nothing)

    while 0xFF & cv2.waitKey(1) != ord('q'):
        
        # Get sliders positions
        ind            = cv2.getTrackbarPos('ind', 'sliders')
        # thresh         = cv2.getTrackbarPos('thresh', 'sliders')
        med_blur_size  = cv2.getTrackbarPos('med_blur_size', 'sliders')
        ksize0         = cv2.getTrackbarPos('ksize'        , 'sliders')
        gauss_size     = cv2.getTrackbarPos('gauss_size'   , 'sliders')
        gauss_size_2   = cv2.getTrackbarPos('gauss_size_2' , 'sliders')

        med_blur_size  = med_blur_size   if med_blur_size % 2 == 1 else med_blur_size + 1
        ksize          = min(ksize0, 31) if ksize0        % 2 == 1 else ksize0 + 1
    
        # Process image
        ref  = images[max(ind-1, 0)].copy()
        gray = images[ind].copy()
        diff = cv2.absdiff(ref, gray)
        
        blur = cv2.medianBlur(gray, med_blur_size)
        gx   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gy   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        g    = cv2.magnitude(gx, gy)
        g    = (255*normalize_image(g)).astype(np.uint8)



        n = gray.size
        gray_norm = gray / (np.sqrt( (1/n)  *  np.sum(gray ** 2)))

        # Compute the Fast Fourier Transform and sum the result:
        fft       = np.fft.fft2(gray_norm)
        
        # Scale and shift the spectrum to improve visualization using the logarithmic transform
        fft       = 20 * np.log(1 + np.abs(fft))
        fft_shift = np.fft.fftshift(fft)
        # fft_shift = normalize_image(fft_shift)

        # fft_shift = fft_shift / (fft_shift.shape[0] * fft_shift.shape[1])

        # Filter the spectrum using a gaussian centered on the middle of the image.
        # This is done to discard the influence of high frequencies.    
        gaussian_pars = (
            fft_shift.shape[1]/2, 
            fft_shift.shape[0]/2, 
            gauss_size, 
            gauss_size) # mx, my, sx, sy

        gaussian_pars_2 = (
            fft_shift.shape[1]/2, 
            fft_shift.shape[0]/2, 
            gauss_size_2, 
            gauss_size_2) # mx, my, sx, sy

        gauss = create_2D_gaussian(fft_shift.shape, *gaussian_pars)
        gauss_2 = create_2D_gaussian(fft_shift.shape, *gaussian_pars_2)
        gauss_3 = gauss - gauss_2

        fft_shift = fft_shift * (
            gauss - gauss_2
        )

        


        spectrum_sum = fft_shift.ravel().sum()

        print(f'spectrum_sum = {spectrum_sum}')





        # blur, _, mag = detect_blur_fft(diff, size=0, verbose=True)
        # spectrum_sum, total_sum, fft_shift, fft_hist = compute_blur(gray, gaussian_sigma = gauss_size, annotate_on_image = True)
        print(f'Spectrum sum: {spectrum_sum}')#\t\t total_sum: {total_sum}')

        val, im_b = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)

        cv2.imshow("gray"      ,        gray)
        cv2.imshow("gray_norm"      ,        normalize_image(gray_norm))
        cv2.imshow("blur"      ,        blur)
        cv2.imshow("fft_shift" ,   normalize_image(fft_shift))
        # cv2.imshow("fft_hist"  ,    fft_hist)
        cv2.imshow("diff"      ,        diff)
        cv2.imshow("mag_grad"  ,           g)
        cv2.imshow("thresh"    ,        im_b)
        cv2.imshow("gauss"     ,     gauss_3)
        # cv2.imshow("ellipses" , result)

    cv2.destroyAllWindows()

def write_list_of_images(list_of_filenames, list_of_images):
    
    for filename, image in zip(list_of_filenames, list_of_images):
        print(f"Saving image file {filename}")
        cv2.imwrite(filename, image)

def print_array_minmax(arr, name=""):
    print(name + f" (min,max) = ({arr.min():.2f}, {arr.max():.2f})")
    return None

def preprocess_list_of_images(images, med_blur_size = 27, ksize = 31):
    
    output_images= []
    for image in images:
        output_image = cv2.medianBlur(image, med_blur_size)
        g = cv2.magnitude(cv2.Sobel(output_image, cv2.CV_64F, 1, 0, ksize=ksize), 
                          cv2.Sobel(output_image, cv2.CV_64F, 0, 1, ksize=ksize))
        g = (255*normalize_image(g)).astype(np.uint8)

        output_images.append(g)
    
    return output_images

def compute_power_spectrum(img):

    N = img.size
    return (1/N**2) * np.sum(np.abs(img.ravel())**2)

def process_ground_truths(name_list, gt_list, out_list, mom_list, config):
    
    imgs = read_list_of_images(name_list)
    ground_thruths = read_list_of_images(gt_list)
    
    areas = []
    moments = []
    for i in range(config['N_IMAGES']):
    
        img = imgs[i]
        gt = ground_thruths[i]
    
        bgr = np.stack((img,)*3, axis=-1)
        contours, hierarchy = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        M = cv2.moments(contours[0])
        
        areas.append(int(M['m00']))
        moments.append(M)
    
        cv2.drawContours(bgr, contours, -1, (0,255,0), 3)
    

        cv2.imwrite(out_list[i], bgr)
    
        with open(mom_list[i], 'w') as fp:
            json.dump(M, fp, sort_keys=True, indent=4)
    
    return (areas, moments)

def compute_correlation(x, y):
    """
    Compute Pearson's correlation between two lists.
    """
    return scipy.stats.pearsonr(x, y)


def compute_particle_density(bead_radius = 10e-6/2, time_decay = 9.06):
    """
    Compute particle density.

    Inputs:     bead_radius is the particle radius in meters.
                time_decay is the time elapsed during the particle fall.
    
    Outputs:    particle_density.

    Defaults:   bead_radius = 10e-6/2
                time_decay  = 9.06

    Comments:   This code is based on:
                Measurement of single leukemia cell's density and mass using optically induced electric field in a microfluidics chip
                Biomicrofluidics 9, 022406 (2015); https://doi.org/10.1063/1.4917290
                Yuliang Zhao1, Hok Sum Sam Lai, Guanglie Zhang, Gwo-Bin Lee, Wen Jung Li
    """

    # r = (10.06e-6)/2     # Beads readius (metros) for 10um particle
    # r = ((26.12)*1e-6)/2 # Beads readius (metros) for 30um particle

    # time_decay = 9.06    # 10um time decay measured
    # time_decay = 3.4629  # 30um time decay measured 

    r = bead_radius

    n = 1.002e-3         # viscosity (Pa*s)= (kg*/s)

    g = 9.82             # Gravity (m/s2)
    a = 4.63             # constante factor (dimensionless)

    sigma = time_decay/a # Time constante (segundos)

    rom = 1.0030e+03     # Medium density (kg/m3)
    term = (9*n)/(2*r*g*sigma)
    roc = rom + term
    
    return roc

def read_csv_files (csv_filenames):
    """
    Compose dataset from a list of '.csv' files.

    Parameters
    ----------
    csv_filenames : list containing csv filenames

    Returns
    -------
    pandas.DataFrame
        A pandas dataframe containing the concatenated .csv files data.

    """

    return pd.concat((pd.read_csv(f, header=None) for f in csv_filenames), axis=1)

def normalize_df_min_max(df):
    """
    Normalize dataframe using column-wise min-max criteria.

    Parameters
    ----------
    df : Input pandas dataframe.

    Returns
    -------
    pandas.DataFrame
        A pandas dataframe with columns normalized to range [0.0, 1.0]
    
    """

    x = df.values #returns a numpy array
    columns = df.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_norm = pd.DataFrame(x_scaled)
    df_norm.columns = columns

    return df_norm


#%% Example to load a scatter plot, perform curve fitting and plot the result
# df = pd.read_csv("../results/Isolada-2-10 um/Isolada-2-10 um_blurscore_size_0067.csv")

# x_data = np.linspace(0, 29, 30)
# y_data = df.to_numpy().ravel()

# pars, cov = fit_single_exponential(x_data, y_data, maxfev=1000)
# a, b = pars
# s1 = np.sqrt(cov[0,0])
# s2 = np.sqrt(cov[1,1])

# x_curve_fit = np.linspace(0, x_data.max(), 1000)
# y_curve_fit = exponential(x_curve_fit, *pars)

# plot_data_and_single_exponential(x_data, y_data, x_curve_fit, y_curve_fit)

#%%
def compute_blur(img, gaussian_sigma = 20, annotate_on_image = False, show_histogram = False, verbose=True):

    # Compute the Fast Fourier Transform and sum the result:
    fft       = np.fft.fft2(img)
    total_sum = np.abs(fft).ravel().sum()
    
    # Scale and shift the spectrum to improve visualization using the logarithmic transform
    fft       = 20 * np.log(1 + np.abs(fft))
    fft_shift = np.fft.fftshift(fft)
    fft_shift = normalize_image(fft_shift)

    # Filter the spectrum using a gaussian centered on the middle of the image.
    # This is done to discard the influence of high frequencies.    
    gaussian_pars = (
        fft_shift.shape[1]/2, 
        fft_shift.shape[0]/2, 
        gaussian_sigma, 
        gaussian_sigma) # mx, my, sx, sy

    fft_shift = fft_shift * create_2D_gaussian(fft_shift.shape, *gaussian_pars)

    # Normalize 
    fft_shift = (255 * normalize_image(fft_shift)).astype('uint8')
    
    spectrum_sum = fft_shift.ravel().sum()

    if show_histogram:
        fft_hist  = compute_histogram_1C(fft_shift)
    else:
        fft_hist = np.zeros(fft_shift.shape, fft_shift.dtype)

    if annotate_on_image:
        fft_shift = cv2.putText(
            fft_shift, f"Max: {np.abs(fft[0,0]):.4e}", 
            (10,10), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA
            )
        fft_shift = cv2.putText(
            fft_shift, f"Sum: {fft_shift.sum():.4e}", 
            (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA
            )
    
    print(f'spectrum_sum = {spectrum_sum}')

    return spectrum_sum, total_sum, fft_shift, fft_hist
