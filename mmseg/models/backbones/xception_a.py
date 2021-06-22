# The original repo: https://github.com/jandylin/DFANet_PyTorch

"""
Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = utils.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""

import math
import torch
import torch.nn as nn

from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.utils import get_root_logger
from ..builder import BACKBONES


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


@BACKBONES.register_module()
class XceptionA(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf

    Modified Xception A architecture, as specified in
    https://arxiv.org/pdf/1904.02216.pdf
    """

    def __init__(self,
                 extra,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False):
        super(XceptionA, self).__init__()

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        self.extra = extra
        self.fc_channels = self.extra['fc_channels']  # should be 1000

        self.conv1 = nn.Conv2d(3, 8, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        # conv for reducing channel size in input for non-first backbone stages
        self.enc2_conv = nn.Conv2d(240, 8, 1, 1, bias=False) # bias=False?

        self.enc2_1 = Block(8, 12, 4, 1, start_with_relu=True, grow_first=True)
        self.enc2_2 = Block(12, 12, 4, 1, start_with_relu=True, grow_first=True)
        self.enc2_3 = Block(12, 48, 4, 2, start_with_relu=True, grow_first=True)
        self.enc2 = nn.Sequential(self.enc2_1, self.enc2_2, self.enc2_3)

        self.enc3_conv = nn.Conv2d(144, 48, 1, 1, bias=False)

        self.enc3_1 = Block(48, 24, 6, 1, start_with_relu=True, grow_first=True)
        self.enc3_2 = Block(24, 24, 6, 1, start_with_relu=True, grow_first=True)
        self.enc3_3 = Block(24, 96, 6, 2, start_with_relu=True, grow_first=True)
        self.enc3 = nn.Sequential(self.enc3_1, self.enc3_2, self.enc3_3)

        self.enc4_conv = nn.Conv2d(288, 96, 1, 1, bias=False)

        self.enc4_1 = Block(96, 48, 4, 1, start_with_relu=True, grow_first=True)
        self.enc4_2 = Block(48, 48, 4, 1, start_with_relu=True, grow_first=True)
        self.enc4_3 = Block(48, 192, 4, 2, start_with_relu=True, grow_first=True)
        self.enc4 = nn.Sequential(self.enc4_1, self.enc4_2, self.enc4_3)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, self.fc_channels)
        self.fca = nn.Conv2d(self.fc_channels, 192, 1)

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
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        enc2 = self.enc2(x)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        pool = self.pooling(enc4)
        fc = self.fc(pool.view(pool.size(0), -1))
        fca = self.fca(fc.view(fc.size(0), -1, 1, 1))
        fca = enc4 * fca

        y_list = enc2, enc3, enc4, fca

        return y_list

    def forward_concat(self, fca_concat, enc2_concat, enc3_concat, enc4_concat):
        """For second and third stage."""

        enc2 = self.enc2(self.enc2_conv(torch.cat((fca_concat, enc2_concat), dim=1)))
        enc3 = self.enc3(self.enc3_conv(torch.cat((enc2, enc3_concat), dim=1)))
        enc4 = self.enc4(self.enc4_conv(torch.cat((enc3, enc4_concat), dim=1)))
        pool = self.pooling(enc4)
        fc = self.fc(pool.view(pool.size(0), -1))
        fca = self.fca(fc.view(fc.size(0), -1, 1, 1))
        fca = enc4 * fca

        y_list = [enc2, enc3, enc4, fca]

        return y_list

    def train(self, mode=True):
        """Convert the model into training mode."""

        super().train(mode)

        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
