"""
"""

import torch
import torch.nn as nn
from .utils import Swish

class ConvBlock(nn.Module):
    """
    """
    def __init__(self, in_channels: int, out_channels: int = None, use_swish: bool = False):
        super(ConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm3d(num_features=out_channels, momentum=0.001, eps=1e-4)
        self.activation = Swish() if use_swish else nn.ReLU()

        self.conv = nn.Sequential(
            self.depthwise_conv, self.pointwise_conv, self.bn, self.activation
        )
    
    def forward(self, x):
        """
        """
        return self.conv(x)


class BiFPN(nn.Module):
    """
    """
    def __init__(self, num_channels, epsilon: int = 1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        
        # Convolutional layers
        self.conv6_up = ConvBlock(num_channels)
        self.conv5_up = ConvBlock(num_channels)
        self.conv4_up = ConvBlock(num_channels)
        self.conv3_up = ConvBlock(num_channels)

        self.conv4_down = ConvBlock(num_channels)
        self.conv5_down = ConvBlock(num_channels)
        self.conv6_down = ConvBlock(num_channels)
        self.conv7_down = ConvBlock(num_channels)

        # Feature Scaling layers
        self.p6_upsample = nn.UpSample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.UpSample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.UpSample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.UpSample(scale_factor=2, mode='nearest')
        
        self.p4_downsample = nn.MaxPool2d(kernel_size=2)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2)
        self.p6_downsample = nn.MaxPool2d(kernel_size=2)
        self.p7_downsample = nn.MaxPool2d(kernel_size=2)

        # Weights
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

    def forward(self, x):
        """
        """

        p3_in, p4_in, p5_in, p6_in, p7_in = x

        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p6_up = self.conv6_up(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in))

        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_up = self.conv5_up(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up))

        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_up = self.conv4_up(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up))

        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv3_up(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up))

        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(
            weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out))
        
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(
            weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out))
        
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_out = self.conv6_down(
            weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out))

        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        p7_out = self.conv7_down(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out))

        return p3_out, p4_out, p5_out, p6_out, p7_out

class Classifier(nn.Module):
    """
    """
    def __init__(self, in_channels, n_anchors, n_classes, n_layers):
        super(Classifier, self).__init__()
        self.n_anchors = n_anchors
        self.n_classes = n_classes
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(
            in_channels, n_anchors * n_classes, kernel_size=3, stride=1, padding=1
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        """
        x = self.layers(x)
        x = self.header(x)
        x = self.activation(x)
        x = x.permute(0, 2, 3, 1)
        out = x.contiguous().view(
            x.shape[0], x.shape[1], x.shape[2], self.n_anchors, self.n_classes
        )
        out = out.contiguous().view(out.shape[0], -1, self.n_classes)
        return out

class Regressor(nn.Module):
    """
    """
    def __init__(self, in_channels, n_anchors, n_layers):
        super(Regressor, self).__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, n_anchors * 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        """
        x = self.layers(x)
        x = self.header(x)
        out = x.permute(0, 2, 3, 1)
        out = out.contiguous().view(out.shape[0], -1, 4)
        return out