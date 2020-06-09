"""
"""

import torch.nn as nn
from efficientnet_pytorch import EfficientNet as Net

class EfficientNet(nn.Module):
    """
    """
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.model = Net.from_pretrained('efficientnet-b0')
        del self.model._conv_head
        del self.model._bn1
        del self.model._avg_pooling
        del self.model._dropout
        del self.model._fc
        self.net = self.model

    def forward(self, x):
        """
        """
        x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))
        feature_maps = []

        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate = drop_connect_rate * float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(x)
        
        return feature_maps[1:]