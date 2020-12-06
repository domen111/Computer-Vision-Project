import cv2 as cv
import time
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', help='images directory')
parser.add_argument('-no', '--no', help='the filename index of the first image', type=int, default=1)
args = parser.parse_args()

if args.dir:
    pathlib.Path(args.dir).mkdir(parents=True, exist_ok=True)

cameras_no = [1, 0]
caps = [cv.VideoCapture(i) for i in cameras_no]
for cap in caps:
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv.CAP_PROP_EXPOSURE, 0)
    # cap.set(3,160)
    # cap.set(4,120)
no = args.no
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
        if not args.dir:
            print('`--dir` missing')
        else:
            cv.imwrite(f'{args.dir}/img{no:04d}l.png', imgs[0])
            cv.imwrite(f'{args.dir}/img{no:04d}r.png', imgs[1])
        no += 1
    elif key == ord('q'):
        break
for cap in caps:
    cap.release()
