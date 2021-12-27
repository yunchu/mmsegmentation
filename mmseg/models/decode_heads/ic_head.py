# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class CascadeFeatureFusion(nn.Module):
    def __init__(self, **kwargs):
        super(CascadeFeatureFusion, self).__init__()
        raise NotImplementedError('Should be re-implemented')

    def forward(self, x_low, x_high):
        raise NotImplementedError('Should be re-implemented')


@HEADS.register_module()
class ICHead(BaseDecodeHead):
    def __init__(self,
                 num_channels=[64, 256, 256],
                 out_channels=[128, 128],
                 **kwargs):
        super(ICHead, self).__init__(**kwargs)
        raise NotImplementedError('Should be re-implemented')

    def forward(self, inputs):
        raise NotImplementedError('Should be re-implemented')

    def losses(self, seg_logit, seg_label):
        raise NotImplementedError('Should be re-implemented')
