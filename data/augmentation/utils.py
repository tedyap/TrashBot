"""
"""

import cv2 as cv
import numpy as np

def draw_rectange(image, coords, color=None):
    """
    """

    im = image.copy()
    coords = coords[:, :4]
    coords = coords.reshape(-1, 4)
    if not color:
        color = [255, 255, 255]
    for coord in coords:
        p1, p2 = (coord[0], coord[1]), (coord[2], coord[3])
        p1 = int(p1[0]), int(p1[1])
        p2 = int(p2[0]), int(p2[1])

        im = cv.rectangle(im.copy(), p1, p2, color, int(max(im.shape[:2]) / 200))
    
    return im

def area(bbox):
    """
    """
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    
def clip_box(bbox, clip_box, alpha):
    """
    """

    area_ = (area(bbox))
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))
    delta = ((area_ - area(bbox)) / area_)

    mask = (delta < (1 - alpha)).astype(int)
    bbox = bbox[mask == 1, :]
    return bbox