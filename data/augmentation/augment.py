"""
"""

import random

from .horizontal import RandomHorizontalFlip  # noqa:F401
from .scale import RandomScale  # noqa:F401
from .rotate import RandomRotate  # noqa:F401
from .translate import RandomTranslate  # noqa:F401

class Pipeline(object):
    """
    """

    def __init__(self, augmentations, prob=1):
        self.augmentations = augmentations
        self.prob = prob
    
    def __call__(self, image, bbox):
        for idx, augmentation in enumerate(self.augmentations):
            if type(self.prob) == list:
                prob = self.prob[idx]
            else:
                prob = self.prob
            
            if random.random() < prob:
                image, bbox = augmentation(image, bbox)
            
        return image, bbox
