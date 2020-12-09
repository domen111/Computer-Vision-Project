from disparity import compute_disparity
import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from pathlib import Path
import argparse


def imshow(myplt, img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        myplt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        myplt.imshow(img)


parser = argparse.ArgumentParser()
parser.add_argument('--display', type=bool, nargs='?', default=False, help='display images')
args = parser.parse_args()


dataset_dir = 'images/cube/'
original_data_path = os.path.join(dataset_dir, 'original data')
depth_data_path = os.path.join(dataset_dir, 'depth data')
post_processing_path = os.path.join(dataset_dir, 'post processing')
display = args.display != False

if display:
    _, ((imgL_ax, imgR_ax), (disparity_ax, imgL_processed_ax)) = plt.subplots(2, 2)
    imgL_ax.set_title('imgL')
    imgR_ax.set_title('imgR')
    disparity_ax.set_title('disparity')
    imgL_processed_ax.set_title('imgL_processed')

for imgL_path in glob.glob(os.path.join(original_data_path, 'img*l.png')):
    img_name = Path(imgL_path).name[:-5]
    imgR_path = os.path.join(original_data_path, img_name + 'r.png')
    print(img_name)

    imgL = cv2.imread(imgL_path)
    imgR = cv2.imread(imgR_path)

    parameters = {
        'numDisparities': 64,
        'blockSize': 5,
        'uniquenessRatio': 0,
        'disp12MaxDiff': 50,
        'speckleWindowSize': 0,
        'speckleRange': 0,
        'autoP1P2': 1,
    }
    disparity = compute_disparity(imgL, imgR, parameters)
    disparity[disparity < 0] = 0
    disparity.dtype = np.uint16
    disparity = cv2.resize(disparity, (imgL.shape[1], imgL.shape[0]))

    imgL_roi = imgL[560:, 550:, :]
    imgR_roi = imgR[560:, 550:, :]
    disparity_roi = disparity[560:, 550:]
    imgL_processed = cv2.imread(os.path.join(post_processing_path, img_name + 'l.png'))
    mask = (imgL_processed == 0).all(axis=2)
    disparity_roi_processed = disparity_roi
    disparity_roi_processed[mask] = 0

    if display:
        disparity_roi_processed_display = disparity_roi
        disparity_roi_processed_display[mask] = np.mean(disparity_roi_processed_display[mask == False])
        imshow(imgL_ax, imgL)
        imshow(imgR_ax, imgR)
        # imshow(disparity_ax, disparity)
        imshow(disparity_ax, disparity_roi_processed_display)
        imshow(imgL_processed_ax, imgL_processed)
        plt.pause(0.1)
        # plt.waitforbuttonpress()

    cv2.imwrite(os.path.join(depth_data_path, img_name + 'disparity.png'), disparity)
    cv2.imwrite(os.path.join(post_processing_path, img_name + 'disparity.png'), disparity_roi_processed)
