"""
Utility functions for dealing with bounding boxes, NMS and Swish activation function.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops.boxes import nms as nms_torch

def nms(det, threshold):
    return nms_torch(det[:, :4], det[:, 4], threshold)

class Swish(nn.Module):
    def forward(self, x):
        """
        """
        return x * torch.sigmoid(x)

class BBoxTransform(nn.Module):
    def __init__(self, mean: int = None, std: int = None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            m = np.array([0, 0, 0, 0]).astype(np.float32)
            self.mean = torch.from_numpy(m)
        else:
            self.mean = mean
        
        if std is None:
            d = np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)
            self.std = torch.from_numpy(d)
        else:
            self.std = std
        
        if torch.cuda.is_available():
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def forward(self, anchors, regressors):
        widths = (anchors[:, :, 2] - anchors[:, :, 0])
        heights = (anchors[:, :, 3] - anchors[:, :, 1])
        center_x = anchors[:, :, 0] + 0.5 * widths
        center_y = anchors[:, :, 1] + 0.5 * heights

        dx = regressors[:, :, 0] * self.std[0] + self.mean[0]
        dy = regressors[:, :, 1] * self.std[1] + self.mean[1]
        dw = regressors[:, :, 2] * self.std[2] + self.mean[2]
        dh = regressors[:, :, 3] * self.std[2] + self.mean[3]

        pred_center_x = center_x + dx * widths
        pred_center_y = center_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_box_x1 = pred_center_x - 0.5 * pred_w
        pred_box_y1 = pred_center_y - 0.5 * pred_h
        pred_box_x2 = pred_center_x + 0.5 * pred_w
        pred_box_y2 = pred_center_y + 0.5 * pred_h

        return torch.stack([pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2], dim=2)

class ClipBoxes(nn.Module):
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, anchors, image):
        _, _, h, w = image.shape
        anchors[:, :, 0] = torch.clamp(anchors[:, :, 0], min=0)
        anchors[:, :, 1] = torch.clamp(anchors[:, :, 1], min=0)
        anchors[:, :, 2] = torch.clamp(anchors[:, :, 2], max=w)
        anchors[:, :, 3] = torch.clamp(anchors[:, :, 3], max=h)

        return anchors

class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        anchors = torch.from_numpy(all_anchors.astype(np.float32))
        if torch.cuda.is_available():
            anchors = anchors.cuda()
        return anchors

def generate_anchors(base=16, ratios=None, scales=None):
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)])
    
    num_anchors = len(ratios) * len(scales)
    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base * np.tile(scales, (2, len(ratios))).T
    area = anchors[:, 2] * anchors[:, 3]
    anchors[:, 2] = np.sqrt(area / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors