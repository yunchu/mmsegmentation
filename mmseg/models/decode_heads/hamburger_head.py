# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from abc import abstractmethod, ABCMeta

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import HEADS
from .decode_head import BaseDecodeHead


class _MatrixDecomposition2DBase(nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError('Should be re-implemented')


class VQ2D(_MatrixDecomposition2DBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError('Should be re-implemented')


class CD2D(_MatrixDecomposition2DBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError('Should be re-implemented')


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError('Should be re-implemented')



def build_ham(key, **kwargs):
    raise NotImplementedError('Should be re-implemented')


class HamburgerV2Plus(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError('Should be re-implemented')

    def forward(self, x):
        raise NotImplementedError('Should be re-implemented')


@HEADS.register_module()
class HamburgerHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError('Should be re-implemented')

    def forward(self, inputs):
        raise NotImplementedError('Should be re-implemented')
