"""
Flipping transformation for data augmentation.
"""

import sys
import os
import random
import numpy as np

path = os.path.join(os.path.relpath("."), "augmentations")
sys.path.append(path)

class RandomHorizontalFlip(object):
    """
    Randomly flip an image with probability p.

    Args:
        p (float): Probability of flipping an image

    Returns:
        numpy.ndarray: Flipped image as a numpy array
        numpy.ndarray: Transformed bounding boxes
    """
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: np.ndarray, bbox: np.ndarray):
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