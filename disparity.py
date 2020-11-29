import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

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


imgL = cv.imread('images/1129-1/l4.png', 0)
imgR = cv.imread('images/1129-1/r4.png', 0)
# imgL = cv.imread('images/im0.jpg', 0)
# imgR = cv.imread('images/im1.jpg', 0)

imgL = downsample_image(imgL, 3)
imgR = downsample_image(imgR, 3)
stereo = cv.StereoBM_create(numDisparities=64, blockSize=31)

# win_size = 5
# min_disp = -1
# max_disp = 63 #min_disp * 9
# num_disp = max_disp - min_disp # Needs to be divisible by 16

# stereo = cv.StereoSGBM_create(minDisparity = min_disp,
#     numDisparities = num_disp,
#     blockSize = 5,
#     uniquenessRatio = 5,
#     speckleWindowSize = 5,
#     speckleRange = 5,
#     disp12MaxDiff = 5,
#     P1 = 8*3*win_size**2,#8*3*win_size**2,
#     P2 = 32*3*win_size**2) #32*3*win_size**2)

disparity = stereo.compute(imgL, imgR)
# disparity[disparity > 0] = 0
cv.imwrite('images/disparity.png', disparity)
print(disparity)
# plt.imshow(imgL)
# plt.imshow(imgR)
# cv.imshow('disparity', disparity)
# cv.waitKey()
plt.imshow(imgL,'gray')
plt.imshow(imgR,'gray')
disparity[disparity < 0] = disparity[disparity > 0].min()
plt.imshow(disparity,'gray')
plt.show()
