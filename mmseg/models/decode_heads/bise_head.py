# Copyright (c) 2018 CoinCheung
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The original repo: https://github.com/CoinCheung/BiSeNet

import math

import torch.nn as nn

from ..builder import HEADS
from ..backbones.bisenet_v2 import ConvBNReLU
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class BiSeHead(BaseDecodeHead):
    def __init__(self,
                 up_factor=8,
                 **kwargs):
        super().__init__(enable_out_seg=False, **kwargs)

        self.conv = ConvBNReLU(self.in_channels, self.channels, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = self.num_classes * up_factor * up_factor
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.channels, out_chan, 1, 1, 0),
            nn.PixelShuffle(up_factor)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        """Forward function."""

        x = self._transform_inputs(inputs)

        y = self.conv(x)
        y = self.drop(y)

        output = self.conv_out(y)

        return output
