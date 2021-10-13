import os
import json

import cv2

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

def write_json (x, json_filename, sort_keys=True, indent=4):
    
    with open(json_filename, 'w') as fp:
        json.dump(x, fp, sort_keys=sort_keys, indent=indent)
    
    return None

def read_json(json_filename):

    with open(json_filename, 'r') as fp:
        x = json.load(fp)
    
    return x

