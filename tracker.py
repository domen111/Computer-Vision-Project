import numpy as np
import cv2 as cv
import cv2
import argparse
import glob
parser = argparse.ArgumentParser(description='This sample demonstrates the meanshift algorithm. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()
# cap = cv.VideoCapture(args.image)
# take first frame of the video
# ret,frame = cap.read()
# setup initial location of window
# x, y, w, h = 300, 200, 100, 50 # simply hardcoded the values
frame = cv2.imread('images/cube/Video-2/img0001l.png')
factor = 8
frame = cv2.resize(frame, (frame.shape[1]//factor, frame.shape[0]//factor))
# track_window = cv2.selectROI('Frame', frame, fromCenter=False, showCrosshair=True)
x, y, w, h = 86, 81, 22, 22
# (x, y, w, h) = track_window
track_window = (x, y, w, h)
# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 100, 1 )
# while(1):
files = sorted(glob.glob('images/cube/Video-2/img*l.png'))
for f in files:
    # ret, frame = cap.read()
    # if ret == True:
    frame = cv2.imread(f)
    frame = cv2.resize(frame, (frame.shape[1]//factor, frame.shape[0]//factor))
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    # apply meanshift to get the new location
    ret, track_window = cv.meanShift(dst, track_window, term_crit)
    # Draw it on image
    x,y,w,h = track_window
    img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
    cv.imshow('img2',img2)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # else:
    #     break
