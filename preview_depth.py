import cv2
from matplotlib import pyplot as plt
from sys import argv


path = argv[1]

img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
img[img == 0] = img[img != 0].mean()
plt.imshow(img)
plt.show()
