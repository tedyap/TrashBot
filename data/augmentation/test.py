"""
"""

import cv2 as cv
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from horizontal import RandomHorizontalFlip
from utils import draw_rectange

# image = cv.imread("messi.jpg")[:,:,::-1]
image = cv.imread("styrofoam cup8.jpg")[:, :, ::-1]
bbox = np.array([[402, 0, 799, 270, 1]]).astype('float64')

# bbox = pkl.load(open("messi_ann.pkl", "rb"))
plt.imshow(draw_rectange(image, bbox))
plt.show()

# print(bbox, type(bbox))

flip = RandomHorizontalFlip(1)
image, bbox = flip(image, bbox)
# print(bbox, type(bbox))

plt.imshow(draw_rectange(image, bbox))
plt.show()