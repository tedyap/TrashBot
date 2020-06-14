"""
"""

import sys
import os
import random
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

path = os.path.join(os.path.relpath("."), "augmentations")
sys.path.append(path)

class RandomHorizontalFlip(object):
    """
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, bbox):
        center = np.array(image.shape[:2])[::-1] / 2
        center = np.hstack((center, center))

        if random.random() < self.prob:
            image = image[:, ::-1, :]
            bbox[:, [0, 2]] += 2 * (
                center[[0, 2]] - bbox[:, [0, 2]]
            )

            w = abs(bbox[:, 0] - bbox[:, 2])
            bbox[:, 0] -= w
            bbox[:, 2] += w
        
        return image, bbox
        
