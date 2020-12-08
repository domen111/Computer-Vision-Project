import cv2

#Function that Downsamples image x number (reduce_factor) of times. 
def downsample_image(image, reduce_factor):
    for i in range(0,reduce_factor):
        #Check if image is color or grayscale
        if len(image.shape) > 2:
            row,col = image.shape[:2]
        else:
            row,col = image.shape

        image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
    return image

def compute_disparity(imgL, imgR, values):
    imgL = downsample_image(imgL, 3)
    imgR = downsample_image(imgR, 3)

    print(values)

    # stereo = cv2.StereoBM_create(numDisparities=values['numDisparities'], \
    #                             blockSize=values['blockSize'])

    stereo = cv2.StereoSGBM_create(
            minDisparity = 0,
            numDisparities = values['numDisparities'],
            blockSize = values['blockSize'],
            uniquenessRatio = values['uniquenessRatio'],
            speckleWindowSize = values['speckleWindowSize'],
            speckleRange = values['speckleRange'],
            disp12MaxDiff = values['disp12MaxDiff'],
            P1 = 8*values['blockSize']**2 if values['autoP1P2'] else values['P1'],
            P2 = 32*values['blockSize']**2 if values['autoP1P2'] else values['P2'],
        )

    disparity = stereo.compute(imgL, imgR)
    return disparity
