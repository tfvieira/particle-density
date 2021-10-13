import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_normalize_img (img, new_shape=(32,32)):

    img = cv2.resize(img, new_shape, interpolation = cv2.INTER_AREA)
    img = (img - img.min())/(img.max() - img.min())

    return img

def subtract_bkg_mean(img, patch_size=5):

    ul  = img[0:patch_size,       0:patch_size]
    bl  = img[-1-patch_size:-1,   0:patch_size]
    ur  = img[0:patch_size,      -1-patch_size:-1]
    br  = img[-1-patch_size:-1,  -1-patch_size:-1]

    print(ul.mean())
    print(bl.mean())
    print(ur.mean())
    print(br.mean())

    bg_avg = np.mean([ul, bl, ur, br])

    return img - bg_avg

def gray2bgr(img):

    return np.stack((img,)*3, axis=-1)

def plot_histogram(img):

    if img.dtype != 'uint8':
        print('ERROR:Image type is not UINT8')
        return -1
    
    # Grayscale image
    if len(img.shape) == 2:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist, color='k')
        plt.show()
    # Color image
    elif len(img.shape) == 3:
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
    plt.show()
    return None

def plot_img_row(img, row=16):

    row = img[row, :]

    plt.stem(range(0, img.shape[1]), row, 'k-')
    plt.show()

    return row

