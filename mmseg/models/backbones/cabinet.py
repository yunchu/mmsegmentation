# Copyright (c) 2020. Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2019 MendelXu
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, constant_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.utils import get_root_logger
from mmseg.models.utils import LocalAttentionModule, PSPModule
from ..builder import BACKBONES
from .mobilenet_v3 import MobileNetV3


class SpatialBranch(nn.Module):
    def __init__(self,
                 in_channels,
                 stem_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU')
        )
        self.conv2 = DepthwiseSeparableConvModule(
            in_channels=stem_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            dw_act_cfg=None,
            pw_act_cfg=dict(type='ReLU')
        )
        self.conv3 = DepthwiseSeparableConvModule(
            in_channels=stem_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            dw_act_cfg=None,
            pw_act_cfg=dict(type='ReLU')
        )
        self.conv_out = ConvModule(
            in_channels=stem_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        out = self.conv_out(y)

        return out


class GhostModule(nn.Module):
    """Reference:
    https://github.com/huawei-noah/CV-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
    License: https://github.com/huawei-noah/CV-Backbones/blob/master/ghostnet_pytorch/License.txt
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 ratio=2,
                 dw_size=3,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvModule(
            in_channels=self.in_channels,
            out_channels=init_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU')
        )
        self.cheap_operation = ConvModule(
            in_channels=init_channels,
            out_channels=new_channels,
            kernel_size=dw_size,
            stride=1,
            padding=dw_size // 2,
            groups=init_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        out = torch.cat([x1, x2], dim=1)
        out = out[:, :self.out_channels, :, :]

        return out


class CAPAttentionModule(nn.Module):
    """Reference: https://github.com/MendelXu/ANN
    """

    def __init__(self,
                 num_channels,
                 key_channels,
                 psp_size=(1, 3, 6, 8),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super().__init__()

        self.num_channels = num_channels
        self.key_channels = key_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.query = ConvModule(
            in_channels=self.num_channels,
            out_channels=self.key_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU')
        )
        self.key = nn.Sequential(
            GhostModule(self.num_channels, self.key_channels, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg),
            PSPModule(psp_size)
        )
        self.value = nn.Sequential(
            GhostModule(self.num_channels, self.num_channels, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg),
            PSPModule(psp_size)
        )

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        key = self.key(x)
        value = self.value(x).permute(0, 2, 1)
        query = self.query(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.num_channels, *x.size()[2:])

        out = x + context

        return out


class ContextBranch(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super().__init__()

        self.in_channels = in_channels
        assert self.in_channels == 3

        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.backbone = MobileNetV3(
            arch='small',
            out_indices=(12,),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
        )
        self.cab = nn.Sequential(
            CAPAttentionModule(
                num_channels=576,
                key_channels=128,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg
            ),
            LocalAttentionModule(
                num_channels=576,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg
            )
        )
        self.conv_out = ConvModule(
            in_channels=576,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x):
        y = self.backbone(x)[0]
        y = self.cab(y)

        out = self.conv_out(y)

        return out


class FeatureFusionModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(FeatureFusionModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv_mix = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU')
        )
        self.conv_sep = DepthwiseSeparableConvModule(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            dw_act_cfg=None,
            pw_act_cfg=dict(type='ReLU')
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels // 4,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels=self.out_channels // 4,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=None
            ),
            nn.Sigmoid()
        )
        self.conv_out = ConvModule(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU')
        )

    def forward(self, fsp, fcp):
        _, _, h, w = fsp.size()
        fcp = F.interpolate(fcp, size=(h, w), mode='bilinear', align_corners=True)

        y = torch.cat([fsp, fcp], dim=1)
        y = self.conv_mix(y)
        y = self.conv_sep(y)

        mask = self.attention(y)
        y = y + mask * y

        out = self.conv_out(y)

        return out


@BACKBONES.register_module()
class CABiNet(nn.Module):
    """CABiNet backbone.

    `Efficient Context Aggregation Network for Low-Latency Semantic Segmentation
    <http://essay.utwente.nl/84370/1/84370_Sasena_Thesis.pdf>`_
    """

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False):
        super().__init__()

        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        self.spatial_branch = SpatialBranch(
            in_channels=in_channels,
            stem_channels=64,
            out_channels=128,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg
        )
        self.context_branch = ContextBranch(
            in_channels=in_channels,
            out_channels=128,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg
        )
        self.ffm = FeatureFusionModule(
            in_channels=256,
            out_channels=128,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg
        )

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""

        spatial = self.spatial_branch(x)
        context = self.context_branch(x)
        fused = self.ffm(spatial, context)

        y_list = [spatial, context, fused]

        return y_list

    def train(self, mode=True):
        """Convert the model into training mode."""

        super().train(mode)

        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
