"""
Translation transformation for data augmentation.
"""

import sys
import os
import random
import numpy as np

from .utils import clip_box

path = os.path.join(os.path.relpath("."), "augmentations")
sys.path.append(path)

class RandomTranslate(object):
    """
    Translate the input image randomly. Bounding boxes with <25% area in
    the transformed image are dropped.

    Args:
        translate (float): Range in (1-translate, 1+translate) randomly chosen as translation factor

    Returns:
        numpy.ndarray: Translated image as a numpy array
        numpy.ndarray: Transformed bounding boxes
    """
    def __init__(self, translate: float = 0.2, diff=False):
        self.translate = translate
        self.diff = diff
    
    def __call__(self, image: np.ndarray, bbox: np.ndarray):
        shape = image.shape
        
        translate_x = random.uniform(*self.translate)
        translate_y = random.uniform(*self.translate)
        
        if not self.diff:
            translate_y = translate_x
            
        canvas = np.zeros(shape).astype(np.uint8)
    
        corner_x = int(translate_x * image.shape[1])
        corner_y = int(translate_y * image.shape[0])
        
        orig_box_coords = [
            max(0, corner_y, max(corner_x, 0), min(shape[0], corner_y + shape[0], min(
                shape[1], corner_x + shape[1])))
        ]
    
        mask = image[
            max(-corner_y, 0):min(shape[0], -corner_y + shape[0]),
            max(-corner_x, 0):min(shape[1], -corner_x + shape[1]), :]
        
        canvas[
            orig_box_coords[0]:orig_box_coords[2],
            orig_box_coords[1]:orig_box_coords[3], :] = mask
        
        image = canvas
        
        bbox[:, :4] += [corner_x, corner_y, corner_x, corner_y]
        bbox = clip_box(bbox, [0, 0, shape[1], shape[0]], 0.25)
        
        return image, bbox

class Translate(object):
    """
    Translate the input image randomly. Bounding boxes with <25% area in
    the transformed image are dropped.

    Args:
        x (float): Transformation factor along X-axis
        y (float): Transformation factor along Y-axis

    Returns:
        numpy.ndarray: Translated image as a numpy array
        numpy.ndarray: Transformed bounding boxes
    """
    def __init__(self, x: float = 0.2, y: float = 0.2, diff=False):
        self.translate__x = x
        self.translate__y = y
        self.diff = diff

    def __call__(self, image: np.ndarray, bbox: np.ndarray):
        shape = image.shape
        translate_x = self.translate__x
        translate_y = self.translate__y

        canvas = np.zeros(shape).astype(np.uint8)
    
        corner_x = int(translate_x * image.shape[1])
        corner_y = int(translate_y * image.shape[0])
        
        orig_box_coords = [
            max(0, corner_y), max(corner_x, 0), min(shape[0], corner_y + shape[0]),
            min(shape[1], corner_x + shape[1])]
    
        mask = image[
            max(-corner_y, 0):min(shape[0], -corner_y + shape[0]),
            max(-corner_x, 0):min(shape[1], -corner_x + shape[1]), :]
        
        canvas[
            orig_box_coords[0]:orig_box_coords[2],
            orig_box_coords[1]:orig_box_coords[3], :] = mask
        
        image = canvas
        
        bbox[:, :4] += [corner_x, corner_y, corner_x, corner_y]
        bbox = clip_box(bbox, [0, 0, shape[1], shape[0]], 0.25)
        
        return image, bbox