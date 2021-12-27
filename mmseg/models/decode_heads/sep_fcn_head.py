# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.cnn import DepthwiseSeparableConvModule

from ..builder import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class DepthwiseSeparableFCNHead(FCNHead):
    """Depthwise-Separable Fully Convolutional Network for Semantic
    Segmentation.

    This head is implemented according to Fast-SCNN paper.
    Args:
        in_channels(int): Number of output channels of FFM.
        channels(int): Number of middle-stage channels in the decode head.
        concat_input(bool): Whether to concatenate original decode input into
            the result of several consecutive convolution layers.
            Default: True.
        num_classes(int): Used to determine the dimension of
            final prediction tensor.
        in_index(int): Correspond with 'out_indices' in FastSCNN backbone.
        norm_cfg (dict | None): Config of norm layers.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_decode(dict): Config of loss type and some
            relevant additional options.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_conv_module(self, in_channels, out_channels, **kwargs):
        return DepthwiseSeparableConvModule(
            in_channels,
            out_channels,
            dw_act_cfg=None,
            **kwargs
        )
