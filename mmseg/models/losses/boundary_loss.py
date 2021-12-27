# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch.nn as nn
from ..builder import LOSSES

def boundary_loss(pred,
                  target,
                  dist,
                  valid_mask,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=255,
                  **kwargs):
    raise NotImplementedError('Should be re-implemented')



def binary_boundary_loss(pred, target, dist, valid_mask):
    raise NotImplementedError('Should be re-implemented')


def one_hot2dist(target):
    raise NotImplementedError('Should be re-implemented')


@LOSSES.register_module()
class BoundaryLoss(nn.Module):
    """BoundaryLoss.

    This loss is proposed in `Boundary loss for highly unbalanced
    segmentation <https://arxiv.org/abs/1812.07032>`.
    """

    def __init__(self,
                 **kwargs):
        super(BoundaryLoss, self).__init__()
        raise NotImplementedError('Should be re-implemented')

    @property
    def name(self):
        return 'boundary'

    def forward(**kwargs):
        raise NotImplementedError('Should be re-implemented')
