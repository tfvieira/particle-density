#%%
import cv2
import numpy as np
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("-o", "--output", help="output filename")

args = parser.parse_args()

if args.verbose:
    print("verbosity turned on")

if args.output:
    # Create image
    img = np.zeros((1,1))

    # Save image
    cv2.imwrite(args.output, img)
