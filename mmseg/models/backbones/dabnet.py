# Copyright (c) 2019 Gen Li
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The original repo: https://github.com/Reagan1311/DABNet

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer, constant_init, normal_init
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.runner import load_checkpoint

from mmseg.utils import get_root_logger
from ..builder import BACKBONES


class Conv(nn.Module):
    def __init__(self, in_channels, num_channels, kernel, stride, padding,
                 dilation=(1, 1), groups=1, bn_act=False, bias=False,
                 conv_cfg=None, norm_cfg=dict(type='BN')):
        super().__init__()

        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            num_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        self.bn_act = bn_act
        if self.bn_act:
            self.bn_prelu = BNPReLU(num_channels, norm_cfg)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_act:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, num_channels, norm_cfg=dict(type='BN')):
        super().__init__()

        self.bn = build_norm_layer(norm_cfg, num_channels)[1]
        self.act = nn.PReLU(num_channels)

    def forward(self, x):
        y = self.bn(x)
        y = self.act(y)

        return y


class DABModule(nn.Module):
    def __init__(self, in_channels, d=1, kernel=3, dilated_kernel=3, conv_cfg=None, norm_cfg=dict(type='BN')):
        super().__init__()

        self.bn_relu_1 = BNPReLU(in_channels, norm_cfg)
        self.conv3x3 = Conv(
            in_channels,
            in_channels // 2,
            kernel=kernel,
            stride=1,
            padding=1,
            bn_act=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )

        self.dconv3x1 = Conv(
            in_channels // 2,
            in_channels // 2,
            kernel=(dilated_kernel, 1),
            stride=1,
            padding=(1, 0),
            groups=in_channels // 2,
            bn_act=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )
        self.dconv1x3 = Conv(
            in_channels // 2,
            in_channels // 2,
            kernel=(1, dilated_kernel),
            stride=1,
            padding=(0, 1),
            groups=in_channels // 2,
            bn_act=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.ddconv3x1 = Conv(
            in_channels // 2,
            in_channels // 2,
            kernel=(dilated_kernel, 1),
            stride=1,
            padding=(1 * d, 0),
            dilation=(d, 1),
            groups=in_channels // 2,
            bn_act=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.ddconv1x3 = Conv(
            in_channels // 2,
            in_channels // 2,
            kernel=(1, dilated_kernel),
            stride=1,
            padding=(0, 1 * d),
            dilation=(1, d),
            groups=in_channels // 2,
            bn_act=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        self.bn_relu_2 = BNPReLU(in_channels // 2, norm_cfg)
        self.conv1x1 = Conv(
            in_channels // 2,
            in_channels,
            kernel=1,
            stride=1,
            padding=0,
            bn_act=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)

        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, num_channels, conv_cfg=None, norm_cfg=dict(type='BN')):
        super().__init__()

        self.in_channels = in_channels
        self.num_channels = num_channels

        if self.in_channels < self.num_channels:
            num_conv = num_channels - in_channels
        else:
            num_conv = num_channels

        self.conv3x3 = Conv(
            in_channels,
            num_conv,
            kernel=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(num_channels, norm_cfg)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.in_channels < self.num_channels:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()

        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input


@BACKBONES.register_module()
class DABNet(nn.Module):
    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False):
        super().__init__()

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        self.extra = extra
        self.block_1 = self.extra['block_1']
        self.block_2 = self.extra['block_2']

        self.init_conv = nn.Sequential(
            Conv(in_channels, 32, 3, 2, padding=1, bn_act=True, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg),
            Conv(32, 32, 3, 1, padding=1, bn_act=True, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg),
            Conv(32, 32, 3, 1, padding=1, bn_act=True, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg),
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times

        self.bn_prelu_1 = BNPReLU(32 + 3, norm_cfg)

        # DAB Block 1
        self.downsample_1 = DownSamplingBlock(32 + 3, 64, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, self.block_1):
            dab_module = DABModule(64, d=2, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), dab_module)
        self.bn_prelu_2 = BNPReLU(128 + 3, norm_cfg)

        # DAB Block 2
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        self.downsample_2 = DownSamplingBlock(128 + 3, 128, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, self.block_2):
            dab_module = DABModule(128, d=dilation_block_2[i], conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i), dab_module)
        self.bn_prelu_3 = BNPReLU(256 + 3, norm_cfg)

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
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, input):
        output0 = self.init_conv(input)

        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))

        # DAB Block 1
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))

        # DAB Block 2
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))

        y_list = [output0_cat, output1_cat, output2_cat]

        return y_list

    def train(self, mode=True):
        """Convert the model into training mode."""

        super().train(mode)

        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
