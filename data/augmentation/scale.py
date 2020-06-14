"""
"""

import sys
import os
import random
import numpy as np
import cv2 as cv

from .utils import clip_box

path = os.path.join(os.path.relpath("."), "augmentations")
sys.path.append(path)

class RandomScale(object):
    """
    """

    def __init__(self, scale=0.2, diff=False):
        self.scale = scale
        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)
        
        self.diff = diff

    def __call__(self, image, bbox):
        shape = image.shape
        if self.diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x

        resize_x = 1 + scale_x
        resize_y = 1 + scale_y

        image = cv.resize(image, None, fx=resize_x, fy=resize_y)
        bbox[:, :4] *= [resize_x, resize_y, resize_x, resize_y]

        canvas = np.zeros(shape, dtype=np.uint8)

        y_lim = int(min(resize_y, 1) * shape[0])
        x_lim = int(min(resize_x, 1) * shape[1])

        canvas[:y_lim, :x_lim, :] = image[:y_lim, :x_lim, :]
        image = canvas
        bbox = clip_box(bbox, [0, 0, 1 + shape[1], shape[0]], .25)

        return image, bbox

class Scale(object):
    """
    """

    def __init__(self, x, y):
        self.scale_x = x
        self.scale_y = y

    def __call__(self, image, bbox):
        shape = image.shape

        resize_x = 1 + self.scale_x
        resize_y = 1 + self.scale_y

        image = cv.resize(image, None, fx=resize_x, fy=resize_y)
        bbox[:, :4] *= [resize_x, resize_y, resize_x, resize_y]

        canvas = np.zeros(shape, dtype=np.uint8)
        y_lim = int(min(resize_y, 1) * shape[0])
        x_lim = int(min(resize_x, 1) * shape[1])

        canvas[:y_lim, :x_lim, :] = image[:y_lim, :x_lim, :]
        image = canvas
        bbox = clip_box(bbox, [0, 0, 1 + shape[1], shape[0]], .25)

        return image, bbox