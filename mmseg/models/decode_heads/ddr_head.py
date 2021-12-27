# Copyright (c) 2020 xingkong
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The original repo: https://github.com/ydhongHIT/DDRNet

import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead

BatchNorm2d = nn.SyncBatchNorm
bn_mom = 0.1


@HEADS.register_module()
class DDRHead(BaseDecodeHead):
    def __init__(self,
                 scale_factor=None,
                 **kwargs):
        super().__init__(enable_out_seg=False, **kwargs)

        self.scale_factor = scale_factor

        self.bn1 = BatchNorm2d(self.in_channels, momentum=bn_mom)
        self.conv1 = nn.Conv2d(self.in_channels, self.channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(self.channels, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.channels, self.num_classes, kernel_size=1, padding=0, bias=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """Forward function."""

        x = self._transform_inputs(inputs)

        y = self.conv1(self.relu(self.bn1(x)))
        output = self.conv2(self.relu(self.bn2(y)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            output = F.interpolate(output, size=[height, width], mode='bilinear')

        return output
