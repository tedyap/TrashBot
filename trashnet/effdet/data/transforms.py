"""
Train time dataset augmentation transforms using numpy.
"""

import numpy as np
import torch
import cv2 as cv

class Normalize(object):
    """
    Normalize input image with ImageNet mean and std dev.

    Args:
        s: Input data
    
    Returns:
        dict containing normalized image and original annotations
    """
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, s):
        image = s['img']

        return {
            'img': ((image.astype(np.float32) - self.mean) / self.std),
            'annot': s['annot']
        }

class Augment(object):
    """
    Augment image during training time. Doesn't save augmented data
    to disk.

    Args:
        flip_x (float): Probability for flipping along X-axis
        flip_y (float): Probability for flipping along Y-axis
        s: Input data

    Returns:

    """
    def __call__(self, s, flip_x=0.5, flip_y=0.5):
        if np.random.rand() < flip_x:
            image, annotation = s['img'], s['annot']
            image = image[:, ::-1, :]
            _, c, _ = image.shape

            x1 = annotation[:, 0].copy()
            x2 = annotation[:, 2].copy()

            temp = x1.copy()
            annotation[:, 0] = c - x2
            annotation[:, 2] = c - temp

            s = {
                'img': image,
                'annot': annotation
            }
        return s

class Resize(object):
    """
    Rezize input image to given dimensions.

    Args:
        size (float): Dimension to resize image
        s: Input data
    
    Returns:
        dict contraining resized image, annotations as numpy arrays
        alongwith original scale
    """
    def __call__(self, s, size=512):
        image, annotation = s['img'], s['annot']
        r, c, _ = image.shape

        if r > c:
            scale = size / r
            resized_r = size
            resized_c = int(c * scale)
        else:
            scale = size / c
            resized_r = int(r * scale)
            resized_c = size
        
        image = cv.resize(image, (resized_c, resized_r))
        new_image = np.zeros((size, size, 3))
        new_image[0:resized_r, 0:resized_c] = image

        annotation[:, :4] *= scale

        return {
            'img': torch.from_numpy(new_image),
            'annot': torch.from_numpy(annotation),
            'scale': scale
        }