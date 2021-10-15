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

    bg_avg = np.mean([ul, bl, ur, br])

    return img - bg_avg


def stack_channel(img, n=3):

    if len(img.shape) > 2:
        print("Warning, stacking image with more than 1 channel.")

    img = np.stack((img,)*n, axis=-1)

    return img


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

def draw_circle(image, center, radius, color=(255,0,0), thickness=1):

    image = cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
    image = (255*image).astype(np.uint8)
    if len(image.shape) == 2:
        image = stack_channel(image, 3)
    
    cv2.circle(image, (int(center[0]), int(center[1])), int(radius), color=color, thickness=thickness)

    return image

def preprocess_image(image, new_shape=(32,32), patch_size=4):

    image = cv2.resize(image, new_shape, interpolation = cv2.INTER_AREA)
    image = subtract_bkg_mean(image)
    image = image/255
    return image