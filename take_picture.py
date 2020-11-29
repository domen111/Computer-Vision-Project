import cv2 as cv
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', help='images dir')
args = parser.parse_args()

cameras_no = [0, 1]
caps = [cv.VideoCapture(i) for i in cameras_no]
for cap in caps:
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv.CAP_PROP_EXPOSURE, 0)
    # cap.set(3,160)
    # cap.set(4,120)
no = 1
while True:
    succ = True
    for cap in caps:
        if cap.grab() == False:
            succ = False
    if succ == False:
        time.sleep(0.1)
        continue
    imgs = [cap.retrieve()[1] for cap in caps]
    cv.imshow('left', imgs[0])
    cv.imshow('right', imgs[1])
    key = cv.waitKey(100)
    if key == ord('s'):
        print(f'save img #{no}')
        cv.imwrite(f'{args.dir}/l{no}.png', imgs[0])
        cv.imwrite(f'{args.dir}/r{no}.png', imgs[1])
        no += 1
    elif key == ord('q'):
        break
for cap in caps:
    cap.release()
