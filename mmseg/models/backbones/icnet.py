# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.ops import resize
from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from ..decode_heads.psp_head import PSPHead
from .resnet import ResNet


@BACKBONES.register_module()
class ICNet(nn.Module):
    def __init__(self,
                 **kwargs):

        super(ICNet, self).__init__()
        raise NotImplementedError('Should be re-implemented')


    def init_weights(self, pretrained=None):
        raise NotImplementedError('Should be re-implemented')

    def forward(self, x):
        raise NotImplementedError('Should be re-implemented')
