import numpy as np
import cv2 as cv
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help='image path')
parser.add_argument('-o', '--output', help='output image path')
parser.add_argument('-p', '--parameters', help='parameters npz file path')
args = parser.parse_args()

board_size = (4, 6)

cv.namedWindow('img', cv.WINDOW_NORMAL)


with np.load(args.parameters) as parameters:
    mtx = parameters['mtx']
    dist = parameters['dist']

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:board_size[1], 0:board_size[0]].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

img = cv.imread(args.image)
# img = cv.resize(img, (504, 378))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, board_size, None)
if ret == True:
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    # Find the rotation and translation vectors.
    # ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
    ret, rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners2, mtx, dist)
    # project 3D points to image plane
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
    img = draw(img, corners2, imgpts)
    cv.drawChessboardCorners(img, board_size, corners2, ret)
    cv.imshow('img', img)
    k = cv.waitKey(0) & 0xFF
    if k == ord('s'):
        cv.imwrite(args.output, img)

cv.destroyAllWindows()

