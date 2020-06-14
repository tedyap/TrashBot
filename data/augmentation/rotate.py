"""
"""

import sys
import os
import random
import numpy as np
import cv2 as cv

from utils import clip_box

path = os.path.join(os.path.relpath("."), "augmentations")
sys.path.append(path)

def rotate_image(image, angle):
    """
    """

    (h, w) = image.shape[:2]
    (cx, cy) = (w // 2, h // 2)

    M = cv.getRotationMatrix2D((cx, cy), angle, 1.)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))

    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy
    
    image = cv.warpAffine(image, M, (nw, nh))
    return image

def get_corners(bbox):
    """
    """
    
    w = (bbox[:, 2] - bbox[:, 0]).reshape(-1, 1)
    h = (bbox[:, 3] - bbox[:, 1]).reshape(-1, 1)

    x1 = bbox[:, 0].reshape(-1, 1)
    y1 = bbox[:, 1].reshape(-1, 1)

    x2 = x1 + w
    y2 = y1

    x3 = x1
    y3 = y1 + h

    x4 = bbox[:, 2].reshape(-1, 1)
    y4 = bbox[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
    return corners

def rotate_box(corners, angle, cx, cy, h, w):
    """
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((
        corners.shape[0], 1), dtype=type(corners[0][0])))
    )
    
    M = cv.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    box = np.dot(M, corners.T).T
    box = box.reshape(-1, 8)
    return box

def get_new_bbox(corners):
    """
    """

    x = corners[:, [0, 2, 4, 6]]
    y = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x, 1).reshape(-1, 1)
    ymin = np.min(y, 1).reshape(-1, 1)
    xmax = np.max(x, 1).reshape(-1, 1)
    ymax = np.max(y, 1).reshape(-1, 1)

    new_bbox = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))
    return new_bbox


class RandomRotate(object):
    """
    """

    def __init__(self, angle=10):
        self.angle = angle
    
    def __call__(self, image, bbox):
        angle = random.uniform(*self.angle)

        w, h = image.shape[1], image.shape[0]
        center_x, center_y = w // 2, h // 2

        image = rotate_image(image, angle)
        
        corners = get_corners(bbox)
        corners = np.hstack((corners, bbox[:, 4:]))
        corners[:, :8] = rotate_box(corners[:, :8], angle, center_x, center_y, h, w)

        new_bbox = get_new_bbox(corners)
        
        scale_x = image.shape[1] / w
        scale_y = image.shape[0] / h

        image = cv.resize(image, (w, h))
        new_bbox[:, :4] /= [scale_x, scale_y, scale_x, scale_y]
        bbox = new_bbox
        bbox = clip_box(bbox, [0, 0, w, h], 0.25)

        return image, bbox