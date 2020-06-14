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