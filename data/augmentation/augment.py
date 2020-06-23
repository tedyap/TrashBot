"""
Pipeline transformations and apply sequentially.
"""

import random
from typing import Union
import numpy as np

from .horizontal import RandomHorizontalFlip  # noqa:F401
from .scale import RandomScale  # noqa:F401
from .rotate import RandomRotate  # noqa:F401
from .translate import RandomTranslate  # noqa:F401

class Pipeline(object):
    """
    Pipeline object for sequentially applying transformations.

    Args:
        augmentations (list): Transformations objects sequentially in a list
        prob (list|int) : Probability of each transformation. If list, each element
                          is probability with which transformation at corresponding
                          index will be applied
    Returns:
        numpy.ndarray: Pipelined input image as a numpy array
        numpy.ndarray: Transformed bounding boxes
    """
    def __init__(self, augmentations: list, prob: Union[list, int] = 1):
        self.augmentations = augmentations
        self.prob = prob
    
    def __call__(self, image: np.ndarray, bbox: np.ndarray):
        for idx, augmentation in enumerate(self.augmentations):
            if type(self.prob) == list:
                prob = self.prob[idx]
            else:
                prob = self.prob
            
            if random.random() < prob:
                image, bbox = augmentation(image, bbox)
            
        return image, bbox