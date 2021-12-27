# Copyright (c) 2021 Juntang Zhuang
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The original repo: https://github.com/juntang-zhuang/ShelfNet

import torch.nn as nn

from ..builder import HEADS
from ..backbones.shelfnet import ConvBNReLU
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class ShelfHead(BaseDecodeHead):
    def __init__(self,
                 **kwargs):
        super().__init__(enable_out_seg=False, **kwargs)

        self.conv = ConvBNReLU(self.in_channels, self.channels, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(self.channels, self.num_classes, kernel_size=3, bias=False, padding=1)

    def init_weights(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, inputs):
        """Forward function."""

        x = self._transform_inputs(inputs)

        y = self.conv(x)
        output = self.conv_out(y)

        return output
