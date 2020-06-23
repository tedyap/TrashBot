"""
Runner script to test during development.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from augment import Pipeline
from horizontal import RandomHorizontalFlip
from scale import RandomScale, Scale
from translate import RandomTranslate, Translate
from rotate import RandomRotate
from utils import draw_rectangle

image = cv.imread("styrofoam cup8.jpg")[:, :, ::-1]
bbox = np.array([[402, 0, 799, 270, 1]]).astype('float64')

plt.imshow(draw_rectangle(image, bbox))
plt.show()

print(bbox, type(bbox))

transforms = Pipeline([RandomHorizontalFlip(1), Translate(x=0.2, y=0.2), RandomRotate((-2, 2))])
image, bbox = transforms(image, bbox)
print(bbox, type(bbox))

plt.imshow(draw_rectangle(image, bbox))
plt.show()