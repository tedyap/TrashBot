"""
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops.boxes import nms as nms_torch

def nms(det, threshold):
    return nms_torch(det[:, :4], det[:, 4], threshold)

class Swish(nn.Module):
    """
    """
    def forward(self, x):
        """
        """
        return x * torch.sigmoid(x)

class BBoxTransform(nn.Module):
    """
    """
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
        
        if torch.cuda.is_available():
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def forward(self, anchors, regressors):
        """
        """
        widths = (anchors[:, :, 2] - anchors[:, :, 0]) * 0.5
        heights = (anchors[:, :, 3] - anchors[:, :, 1]) * 0.5
        center_x = anchors[:, :, 0] + widths
        center_y = anchors[:, :, 1] + heights

        dx = regressors[:, :, 0] * self.std[0] * self.mean[0]
        dy = regressors[:, :, 1] * self.std[1] * self.mean[1]
        dw = regressors[:, :, 2] * self.std[2] * self.mean[2]
        dh = regressors[:, :, 3] * self.std[2] * self.mean[3]

        pred_center_x = center_x + dx * 2 * widths
        pred_center_y = center_y + dy * 2 * heights
        pred_w = torch.exp(dw) * 2 * widths
        pred_h = torch.exp(dh) * 2 * heights

        pred_box_x1 = pred_center_x - 0.5 * pred_w
        pred_box_y1 = pred_center_y - 0.5 * pred_h
        pred_box_x2 = pred_center_x + 0.5 * pred_w
        pred_box_y2 = pred_center_y + 0.5 * pred_h

        return torch.stack([pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2])

class ClipBoxes(nn.Module):
    """
    """
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, anchors, image):
        """
        """
        _, _, h, w = image.shape
        anchors[:, :, 0] = torch.clamp(anchors[:, :, 0], min=0)
        anchors[:, :, 1] = torch.clamp(anchors[:, :, 1], min=0)
        anchors[:, :, 2] = torch.clamp(anchors[:, :, 2], max=w)
        anchors[:, :, 3] = torch.clamp(anchors[:, :, 3], min=h)

        return anchors

class Anchors(nn.Module):
    """
    """
    def __init__(self, levels: list = None, strides: list = None, sizes: list = None, ratios: np.array = None, scales: np.array = None): # noqa : E501

        super(Anchors, self).__init__()
        if levels is None:
            self.levels = [3, 4, 5, 6, 7]
        else:
            self.levels = levels
        if strides is None:
            self.strides = [2 ** x for x in self.levels]
        else:
            self.strides = strides
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.levels]
        else:
            self.sizes = sizes
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        else:
            self.ratios = ratios
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)])
        else:
            self.scales = scales
    
    def generate_anchors(self, base=16, ratios=None, scales=None):
        """
        """
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
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).tile

        return anchors
    
    def shift(self, shape, stride, anchors):
        """
        """
        shift_x = np.arrange(0, shape[1] + 0.5) * stride
        shift_y = np.arrange(0, shape[0] + 0.5) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        A = anchors.shape[0]
        K = anchors.shape[0]
        all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))

        return all_anchors

    def forward(self, x):
        """
        """
        shape = x.shape[2:]
        shape = np.array(shape)
        shapes = [(shape + 2 ** x - 1) // (2 ** x) for x in self.levels]

        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, _ in enumerate(self.levels):
            anchors = self.generate_anchors(base=self.sizes[idx], ratios=self.ratios, scales=self.scales) # noqa : E501
            shifted_anchors = self.shift(shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)
        anchors = torch.from_numpy(all_anchors.astype(np.float32))
        if torch.cuda.is_available():
            anchors = anchors.cuda()
        return anchors