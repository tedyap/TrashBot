"""
"""

import numpy as np
import torch
import torch.nn as nn

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