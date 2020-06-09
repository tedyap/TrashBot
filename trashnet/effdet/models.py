"""
"""

import torch
import torch.nn as nn
from effnet import EfficientNet
from layers import ConvBlock, BiFPN, Regressor, Classifier
from utils import ClipBoxes, BBoxTransform
from losses import FocalLoss

class EfficientDet(nn.Module):
    """
    """
    def __init__(self, n_anchors: int =9, n_classes: int =20, compound_coef: int = 0):
        super(EfficientDet, self).__init__()
        self.backbone = EfficientNet()
        self.compound_coef = compound_coef
        self.n_channels = [64, 88, 112, 160, 224, 288, 384, 384][self.compound_coef]

        self.conv3 = nn.Conv2d(40, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(80, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(192, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(192, self.n_channels, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=2, padding=1)
        )
        self.bifpn = nn.Sequential(*[BiFPN(self.n_channels) for i in range(
            min(2 + self.compound_coef, 8))]
        )
        self.n_classes = n_classes
        
        self.regressor = Regressor(
            in_channels=self.n_channels, n_anchors=n_anchors,
            n_layers=3 + self.compound_coef // 3
        )

        self.classifier = Classifier(
            in_channels=self.n_channels, n_anchors=self.n_anchors,
            n_classes=self.n_classes, n_layers=3 + self.compound_coef // 3
        )
        self.clips = ClipBoxes()
        self.loss = FocalLoss()
        self.regress = BBoxTransform()
        



