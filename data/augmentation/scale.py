"""
Scale transformation for data augmentation.
"""

import sys
import os
import random
import numpy as np
import cv2 as cv
from typing import Union

from .utils import clip_box

path = os.path.join(os.path.relpath("."), "augmentations")
sys.path.append(path)

class RandomScale(object):
    """
    Randomly scale an image. Bounding boxes with <25% of area in
    the transformed image are dropped.

    Args:
        scale (tuple|float): Range in (1-translate, 1+translate) randomly chosen as scale factor
                            If tuple, scale is randomly chosen from the range from the tuple

    Returns:
        numpy.ndarray: Scaled image as a numpy array
        numpy.ndarray: Transformed bounding boxes
    """
    def __init__(self, scale: Union[tuple, float] = 0.2, diff=False):
        self.scale = scale
        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)
        
        self.diff = diff

    def __call__(self, image: np.ndarray, bbox: np.ndarray):
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
    Scale an image. Bounding boxes with <25% of area in
    the transformed image are dropped.

    Args:
        x (float): Scaling factor along X-axis
        y (float): Scaling factor along Y-axis

    Returns:
        numpy.ndarray: Scaled image as a numpy array
        numpy.ndarray: Transformed bounding boxes
    """
    def __init__(self, x: float, y: float):
        self.scale_x = x
        self.scale_y = y

    def __call__(self, image: np.ndarray, bbox: np.ndarray):
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