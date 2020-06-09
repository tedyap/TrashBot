"""
"""

import math

import torch
import torch.nn as nn

from effnet import EfficientNet
from layers import ConvBlock, BiFPN, Regressor, Classifier
from utils import ClipBoxes, BBoxTransform, Anchors, nms
from losses import FocalLoss

class EfficientDet(nn.Module):
    """
    """
    def __init__(self, n_anchors: int = 9, n_classes: int = 20, compound_coef: int = 0):
        super(EfficientDet, self).__init__()
        self.backbone = EfficientNet()
        self.prior = 0.01
        self.classification_threshold = 0.05
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
        self.anchors = Anchors()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m._out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.regressor.header.weight.data.fill_(0)
        self.regressor.header.bias.data.fill_(0)

        self.classifier.header.weight.data.fill_(0)
        self.classifier.header.bias.data.fill_(-math.log((1. - self.prior) / self.prior))        

    def freeze(self):
        """
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def forward(self, x):
        """
        """
        if len(x) > 1:
            train = True
            batch, annotations = x
        else:
            train = False
            batch = x

        c3, c4, c5 = self.backbone(batch)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)
        p6 = self.conv6(c5)
        p7 = self.conv7(p6)

        feature_maps = [p3, p4, p5, p6, p7]
        features = self.bifpn(feature_maps)

        regression = torch.cat([self.regressor(feature) for feature in features], dim=1)
        classification = torch.cat([self.classifier(feature) for feature in features], dim=1)
        anchors = self.anchors(batch)

        if train:
            return self.loss(classification, regression, anchors, annotations)

        transformed_anchors = self.regress(anchors, regression)
        transformed_anchors = self.clips(transformed_anchors, batch)
        score = torch.max(classification, dim=2, keepdim=True)[0]
        threshold_score = (score > self.classification_threshold)[0, :, 0]

        if threshold_score.sum() == 0:
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

        classification = classification[:, threshold_score, :]
        transformed_anchors = transformed_anchors[:, threshold_score, :]
        score = score[:, threshold_score, :]

        nms_idx = nms(torch.cat([transformed_anchors, score], dim=2)[0, :, :], 0.6)
        nms_scores, nms_class = classification[0, nms_idx, :].max(dim=1)
        
        return [nms_scores, nms_class, transformed_anchors[0, nms_idx, :]]