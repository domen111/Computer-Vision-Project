from disparity import compute_disparity
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import sys

dir = 'images/cube/original data'
no = 149
imgL = cv2.imread(f'{dir}/img{no:04d}l.png', cv2.IMREAD_GRAYSCALE)
imgL_color = cv2.imread(f'{dir}/img{no:04d}l.png')
imgR = cv2.imread(f'{dir}/img{no:04d}r.png', cv2.IMREAD_GRAYSCALE)

fig = plt.figure()
fig.canvas.mpl_connect('close_event', lambda e: sys.exit())
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.5)
ax.imshow(cv2.cvtColor(((imgL.astype(float) + imgR.astype(float)) / 2).astype('uint8'),
                      cv2.COLOR_GRAY2RGB))


slider_vars = [
    {'name': 'plotMinValue', 'default': -1, 'min': -1, 'max': 1000, 'step': 1},
    {'name': 'plotMaxValue', 'default': -1, 'min': -1, 'max': 1000, 'step': 1},
    {'name': 'imgType', 'default': 0, 'min': 0, 'max': 2, 'step': 1},
    {'name': 'blurSize', 'default': 0, 'min': 0, 'max': 400, 'step': 1},
    {'name': 'swapLR', 'default': 0, 'min': 0, 'max': 1, 'step': 1},
    {'name': 'numDisparities', 'default': 64, 'min': 16, 'max': 160, 'step': 16},
    {'name': 'blockSize', 'default': 3, 'min': 1, 'max': 101, 'step': 2},
    {'name': 'uniquenessRatio', 'default': 0, 'min': 0, 'max': 15, 'step': 1},
    {'name': 'disp12MaxDiff', 'default': 30, 'min': 0, 'max': 200, 'step': 1},
    {'name': 'speckleWindowSize', 'default': 0, 'min': 0, 'max': 100, 'step': 1},
    {'name': 'speckleRange', 'default': 0, 'min': 0, 'max': 15, 'step': 1},
    {'name': 'autoP1P2', 'default': 1, 'min': 0, 'max': 1, 'step': 1},
    {'name': 'P1', 'default': 0, 'min': 0, 'max': 500, 'step': 1},
    {'name': 'P2', 'default': 0, 'min': 0, 'max': 500, 'step': 1},
]
sliders_ax = []
sliders = []
for i, var in enumerate(slider_vars):
    slider_ax = fig.add_axes([0.25, 0.1 + (len(slider_vars) - i - 1) * 0.03, 0.65, 0.02])
    sliders_ax.append(sliders_ax)
    slider = Slider(slider_ax, var['name'], var['min'], var['max'], var['default'], valstep=var['step'])
    sliders.append(slider)
# plt.pause(3)

last_value = dict()
while True:
    values = {var['name']: sliders[i].val for i, var in enumerate(slider_vars)}
    if values != last_value:
        last_value = values
        tmpL, tmpR = (imgR, imgL) if values['swapLR'] else (imgL, imgR)
        imgType = values['imgType']
        if imgType == 0:
            print(values)
            disparity = compute_disparity(tmpL, tmpR, values)
            if values['plotMinValue'] != -1:
                disparity[disparity < values['plotMinValue']] = values['plotMinValue']
            if values['plotMaxValue'] != -1:
                disparity[disparity > values['plotMaxValue']] = values['plotMaxValue']
            ax.imshow(disparity)
        elif imgType == 1:
            imgL_color_blur = cv2.blur(imgL_color,(values['blurSize'],values['blurSize']),0)  if values['blurSize'] > 0 else imgL_color
            hsv = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2HSV)
            rgb = cv2.cvtColor(imgL_color_blur, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
        elif imgType == 2:
            imgL_color_blur = cv2.blur(imgL_color,(values['blurSize'],values['blurSize']),0) if values['blurSize'] > 0 else imgL_color
            # tmp = np.expand_dims(np.mean(imgL_color_blur, axis=2), axis=2)
            # tmp = np.sum(imgL_color_blur - np.concatenate((tmp, tmp, tmp), axis=2), axis=2)
            hsv = cv2.cvtColor(imgL_color_blur, cv2.COLOR_BGR2HSV)
            # hsv[hsv[:,:,2] < 10] = 0
            res = hsv[:,:,1]
            # ax.imshow(tmp)
            if values['plotMinValue'] != -1:
                res[res < values['plotMinValue']] = 0
            if values['plotMaxValue'] != -1:
                res[res > values['plotMaxValue']] = 0
            ax.imshow(res)
    plt.pause(0.3)
