import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt

imgL = cv.imread('images/stereo 11-1/iphone 8/IMG_1978.jpeg', 0)
imgR = cv.imread('images/stereo 11-1/iphone se/IMG_0439.jpeg', 0)

imgL = cv.resize(imgL, (378, 504))
imgR = cv.resize(imgR, (378, 504))
stereo = cv.StereoBM_create(numDisparities=24, blockSize=15)
disparity = stereo.compute(imgL,imgR)
cv.imwrite('disparity.jpeg', disparity)
# plt.imshow(disparity,'gray')
# plt.show()
