import numpy as np
import cv2 as cv
import glob
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help='image path')
parser.add_argument('-o', '--output', help='output image path')
parser.add_argument('-p', '--parameters', help='parameters npz file path')
args = parser.parse_args()


with np.load(args.parameters) as parameters:
    mtx = parameters['mtx']
    dist = parameters['dist']

img = cv.imread(args.image)
# img = cv.resize(img, (504, 378))
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv.imwrite(args.output, dst)