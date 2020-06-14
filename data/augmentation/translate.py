""" # noqa : E902
"""

import sys
import os
import random
import numpy as np
import cv2 as cv

from utils import clip_box

path = os.path.join(os.path.relpath("."), "augmentations")
sys.path.append(path)

class RandomTranslate(object):
    """
    """

    def __init__(self, translate=0.2, diff=False):
        self.translate = translate
        self.diff = diff
    
    def __call__(self, image, bbox):
        shape = image.shape
        
        translate__x = random.uniform(*self.translate)
        translate__y = random.uniform(*self.translate)
        
        if not self.diff:
            translate_y = translate_x
            
        canvas = np.zeros(shape).astype(np.uint8)
    
        corner_x = int(translate__x * image.shape[1])
        corner_y = int(translate__y * image.shape[0])
        
        orig_box_coords = [
            max(0, corner_y, max(corner_x, 0), min(shape[0], corner_y + shape[0],
            min(shape[1], corner_x + shape[1]))
        ]
    
        mask = image[
            max(-corner_y, 0):min(shape[0], -corner_y + shape[0]),
            max(-corner_x, 0):min(shape[1], -corner_x + shape[1]),
            :]
        
        canvas[
            orig_box_coords[0]:orig_box_coords[2],
            orig_box_coords[1]:orig_box_coords[3],
        :] = mask
        
        image = canvas
        
        bbox[:,:4] += [corner_x, corner_y, corner_x, corner_y]
        bbox = clip_box(bbox, [0, 0, shape[1], shape[0]], 0.25)
        
        return image, bbox

class Translate(object):
    """
    """

    def __init__(self, x=0.2, y=0.2, diff=False):
        self.translate__x = x
        self.translate__y = y
        self.diff = diff

    def __call__(self, image, bbox):
        shape = image.shape

        translate_x = self.translate__x
        translate_y = self.translate__y

        canvas = np.zeros(shape).astype(np.uint8)
    
        corner_x = int(translate__x * image.shape[1])
        corner_y = int(translate__y * image.shape[0])
        
        orig_box_coords = [
            max(0, corner_y, max(corner_x, 0), min(shape[0], corner_y + shape[0],
            min(shape[1], corner_x + shape[1]))
        ]
    
        mask = image[
            max(-corner_y, 0):min(shape[0], -corner_y + shape[0]),
            max(-corner_x, 0):min(shape[1], -corner_x + shape[1]),
            :]
        
        canvas[
            orig_box_coords[0]:orig_box_coords[2],
            orig_box_coords[1]:orig_box_coords[3],
        :] = mask
        
        image = canvas
        
        bbox[:,:4] += [corner_x, corner_y, corner_x, corner_y]
        bbox = clip_box(bbox, [0, 0, shape[1], shape[0]], 0.25)
        
        return image, bbox