import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import time
import sys

#Function that Downsamples image x number (reduce_factor) of times. 
def downsample_image(image, reduce_factor):
    for i in range(0,reduce_factor):
        #Check if image is color or grayscale
        if len(image.shape) > 2:
            row,col = image.shape[:2]
        else:
            row,col = image.shape

        image = cv.pyrDown(image, dstsize= (col//2, row // 2))
    return image

def compute_disparity(imgL, imgR, values):
    imgL = downsample_image(imgL, 3)
    imgR = downsample_image(imgR, 3)

    # numDisparities = cv.getTrackbarPos('numDisparities', 'disparity') * 16
    # blockSize = cv.getTrackbarPos('blockSize', 'disparity') * 2 + 1
    numDisparities = 64
    blockSize = 15

    print(values)
    stereo = cv.StereoBM_create(numDisparities=values['numDisparities'], \
                                blockSize=values['blockSize'])

    # win_size = 5
    # min_disp = -1
    # max_disp = 63 #min_disp * 9
    # num_disp = max_disp - min_disp # Needs to be divisible by 16

    # stereo = cv.StereoSGBM_create(minDisparity = min_disp,
    #     numDisparities = num_disp,
    #     blockSize = 31,
    #     uniquenessRatio = 5,
    #     speckleWindowSize = 5,
    #     speckleRange = 5,
    #     disp12MaxDiff = 5,
    #     P1 = 8*3*win_size**2,#8*3*win_size**2,
    #     P2 = 32*3*win_size**2) #32*3*win_size**2)

    disparity = stereo.compute(imgL, imgR)
    return disparity
    # disparity[disparity > 0] = 0
    cv.imwrite('images/disparity.png', disparity)
    print(disparity)
    # plt.imshow(imgL)
    # plt.imshow(imgR)
    cv.imshow('disparity', disparity)
    cv.waitKey()
    # plt.imshow(imgL,'gray')
    # plt.imshow(imgR,'gray')
    # disparity[disparity < 0] = disparity[disparity > 0].min()
    # plt.imshow(disparity,'gray')
    # plt.show()

def trackerbar_event(val):
    pass


imgL = cv.imread('images/1129-1/l4.png', 0)
imgR = cv.imread('images/1129-1/r4.png', 0)
# imgL = cv.imread('images/im0.jpg', 0)
# imgR = cv.imread('images/im1.jpg', 0)

# cv.namedWindow('disparity', cv.WINDOW_AUTOSIZE | cv.WINDOW_GUI_EXPANDED)
# cv.createTrackbar('numDisparities', 'disparity', 64 // 16, 200 // 16, trackerbar_event)
# cv.createTrackbar('blockSize', 'disparity', 30 // 2, 100 // 2, trackerbar_event)

fig = plt.figure()
fig.canvas.mpl_connect('close_event', lambda e: sys.exit())
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.25)
ax.imshow(imgL)


slider_vars = [
    {'name': 'numDisparities', 'default': 64, 'min': 16, 'max': 160, 'step': 16},
    {'name': 'blockSize', 'default': 15, 'min': 1, 'max': 101, 'step': 2},
]
sliders_ax = []
sliders = []
for i, var in enumerate(slider_vars):
    # _, ax = plt.subplots()
    # plt.subplots_adjust(bottom=0.2,left=0.3)
    # ax = plt.axes([0.25, 0.1, 1, 0.03]) if i == 0 else plt.axes()
    slider_ax = fig.add_axes([0.25, 0.1 + i * 0.05, 0.65, 0.03])
    sliders_ax.append(sliders_ax)
    slider = Slider(slider_ax, var['name'], var['min'], var['max'], var['default'], valstep=var['step'])
    sliders.append(slider)
# plt.show()
plt.pause(3)

# h = plt.imshow(imgL, 'gray')
while True:
    disparity = compute_disparity(imgL, imgR, \
        {var['name']: sliders[i].val for i, var in enumerate(slider_vars)})
    # disparity[disparity < 0] = disparity[disparity > 0].min()
    # cv.imshow('disparity', disparity)
    # disparity[disparity < 0] = disparity[disparity > 0].min()
    ax.imshow(disparity)
    # plt.draw()
    plt.pause(1)
    # time.sleep(5)
    # cv.waitKey(500)
