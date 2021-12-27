# Copyright (C) 2020-2021 YuvalNirkin
# SPDX-License-Identifier: CC0-1.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The original repo: https://github.com/YuvalNirkin/hyperseg

import numbers
from itertools import groupby

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from mmcv.cnn import build_norm_layer, build_activation_layer

from mmseg.core import normalize
from mmseg.models.utils import MetaConv2d, MetaSequential
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class MultiScaleDecoder(nn.Module):
    """ Dynamic multi-scale decoder.

    Args:
        feat_channels (list of int): per level input feature channels.
        signal_channels (list of int): per level input signal channels.
        num_classes (int): output number of classes.
        kernel_sizes (int): the kernel size of the layers.
        level_layers (int): number of layers in each level.
        level_channels (list of int, optional): If specified, sets the output channels of each level.
        norm_cfg (dict): Type of feature normalization layer
        act_cfg (dict): Type of activation layer
        out_kernel_size (int): kernel size of the final output layer.
        expand_ratio (int): inverted residual block's expansion ratio.
        groups (int, optional): number of blocked connections from input channels to output channels.
        weight_groups (int, optional): per level signal to weights.
        with_out_fc (bool): If True, add a final fully connected layer.
        dropout (float): If specified, enables dropout with the given probability.
        coords_res (list of tuple of int, optional): list of inference resolutions for caching positional embedding.
        unify_level (int, optional): the starting level to unify the signal to weights operation from.
    """

    def __init__(self, feat_channels, signal_channels, num_classes=3, kernel_sizes=3, level_layers=1,
                 level_channels=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU6'),
                 out_kernel_size=1, expand_ratio=1, groups=1, weight_groups=1, with_out_fc=False,
                 with_out_norm=False, dropout=None, coords_res=None, unify_level=None):  # must be a list of tuples
        super().__init__()

        if isinstance(kernel_sizes, numbers.Number):
            kernel_sizes = (kernel_sizes,) * len(level_channels)
        if isinstance(level_layers, numbers.Number):
            level_layers = (level_layers,) * len(level_channels)
        if isinstance(expand_ratio, numbers.Number):
            expand_ratio = (expand_ratio,) * len(level_channels)
        assert len(kernel_sizes) == len(level_channels), \
            f'kernel_sizes ({len(kernel_sizes)}) must be of size {len(level_channels)}'
        assert len(level_layers) == len(level_channels), \
            f'level_layers ({len(level_layers)}) must be of size {len(level_channels)}'
        assert len(expand_ratio) == len(level_channels), \
            f'expand_ratio ({len(expand_ratio)}) must be of size {len(level_channels)}'
        if isinstance(groups, (list, tuple)):
            assert len(groups) == len(level_channels), f'groups ({len(groups)}) must be of size {len(level_channels)}'

        self.level_layers = level_layers
        self.levels = len(level_channels)
        self.unify_level = unify_level
        self.layer_params = []
        feat_channels = feat_channels[::-1]  # Reverse the order of the feature channels
        self.coords_cache = {}
        self.weight_groups = weight_groups
        self.level_blocks = nn.ModuleList()
        self.weight_blocks = nn.ModuleList()
        self._ranges = [0]

        # For each level
        prev_channels = 0
        for level in range(self.levels):
            curr_ngf = feat_channels[level]
            curr_out_ngf = curr_ngf if level_channels is None else level_channels[level]
            prev_channels += curr_ngf  # Accommodate the previous number of channels
            kernel_size = kernel_sizes[level]

            # For each layer in the current level
            curr_layers = []
            for layer in range(self.level_layers[level]):
                if (not with_out_fc) and (level == (self.levels - 1) and (layer == (self.level_layers[level] - 1))):
                    curr_out_ngf = num_classes

                if kernel_size > 1:
                    curr_layers.append(HyperPatchInvertedResidual(
                        prev_channels + 2,
                        curr_out_ngf,
                        kernel_size,
                        expand_ratio=expand_ratio[level],
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg
                    ))
                else:
                    group = groups[level] if isinstance(groups, (list, tuple)) else groups
                    curr_layers.append(make_hyper_patch_conv2d_block(
                        prev_channels + 2,
                        curr_out_ngf,
                        kernel_size,
                        groups=group,
                        norm_cfg=norm_cfg
                    ))
                prev_channels = curr_out_ngf

            self.level_blocks.append(MetaSequential(*curr_layers))

            if level < (unify_level - 1):
                self.weight_blocks.append(WeightLayer(
                    self.level_blocks[-1].hyper_params
                ))
            else:
                self._ranges.append(self._ranges[-1] + self.level_blocks[-1].hyper_params)
                if level == (self.levels - 1):
                    hyper_params = sum([b.hyper_params for b in self.level_blocks[unify_level - 1:]])
                    self.weight_blocks.append(WeightLayer(
                        hyper_params
                    ))

        self.with_out_norm = with_out_norm
        if self.with_out_norm:
            assert with_out_fc

        # Add the last layer
        if with_out_fc:
            out_fc_layers = [nn.Dropout2d(dropout, True)] if dropout is not None else []
            out_fc_layers.append(HyperPatchConv2d(
                prev_channels,
                num_classes,
                out_kernel_size,
                padding=out_kernel_size // 2,
                norm_weights=self.with_out_norm
            ))
            self.out_fc = MetaSequential(*out_fc_layers)
        else:
            self.out_fc = None

        # Cache image coordinates
        if coords_res is not None:
            for res in coords_res:
                res_pyd = [(res[0] // 2 ** i, res[1] // 2 ** i) for i in range(self.levels)]
                for level_res in res_pyd:
                    self.register_buffer(
                        f'coord{level_res[0]}_{level_res[1]}',
                        self.cache_image_coordinates(*level_res)
                    )

        # Initialize signal to weights
        self.param_groups = get_hyper_params(self)
        min_unit = max(weight_groups)
        signal_features = divide_feature(signal_channels, self.param_groups, min_unit=min_unit)
        init_signal2weights(self, list(signal_features), weight_groups=weight_groups)
        self.hyper_params = sum(self.param_groups)

    @staticmethod
    def cache_image_coordinates(h, w):
        x = torch.linspace(-1, 1, steps=w)
        y = torch.linspace(-1, 1, steps=h)
        grid = torch.stack(torch.meshgrid(y, x)[::-1], dim=0).unsqueeze(0)

        return grid

    def get_image_coordinates(self, b, h, w, device):
        cache = f'coord{h}_{w}'
        if hasattr(self, cache):
            return getattr(self, cache).expand(b, -1, -1, -1)

        x = torch.linspace(-1, 1, steps=w, device=device)
        y = torch.linspace(-1, 1, steps=h, device=device)
        grid = torch.stack(torch.meshgrid(y, x)[::-1], dim=0).unsqueeze(0)

        return grid.expand(b, -1, -1, -1)

    def forward(self, x, s):
        # For each level
        p = None
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            weight_block = self.weight_blocks[min(level, self.unify_level - 1)]

            # Initial layer input
            if p is None:
                p = x[-level - 1]
            else:
                if p.shape[2:] != x[-level - 1].shape[2:]:
                    p = F.interpolate(p, x[-level - 1].shape[2:], mode='bilinear', align_corners=False)  # Upsample
                p = torch.cat((x[-level - 1], p), dim=1)

            # Add image coordinates
            p = torch.cat([self.get_image_coordinates(p.shape[0], *p.shape[-2:], p.device), p], dim=1)

            # Compute the output for the current level
            if level < self.unify_level - 1:
                w = weight_block(s)
                p = level_block(p, w)
            else:
                if level == self.unify_level - 1:
                    w = weight_block(s)
                i = level - self.unify_level + 1
                p = level_block(p, w[:, self._ranges[i]:self._ranges[i + 1]])

        if self.with_out_norm:
            p = normalize(p, dim=1, p=2)

        # Last layer
        if self.out_fc is not None:
            p = self.out_fc(p, s)

        # Upscale the prediction the finest feature map resolution
        if p.shape[2:] != x[0].shape[2:]:
            p = F.interpolate(p, x[0].shape[2:], mode='bilinear', align_corners=False)  # Upsample

        return p


def get_hyper_params(model):
    hyper_params = []

    # For each child module
    for name, m in model.named_children():
        if isinstance(m, (WeightLayer,)):
            hyper_params.append(m.target_params)
        else:
            hyper_params += get_hyper_params(m)

    return hyper_params


def init_signal2weights(model, signal_features, signal_index=0, weight_groups=1):
    # For each child module
    for name, m in model.named_children():
        if isinstance(m, (WeightLayer,)):
            curr_feature_nc = signal_features.pop(0)
            curr_weight_group = weight_groups.pop(0) if isinstance(weight_groups, list) else weight_groups
            m.init_signal2weights(curr_feature_nc, signal_index, curr_weight_group)
            signal_index += curr_feature_nc
        else:
            init_signal2weights(m, signal_features, signal_index, weight_groups)


class WeightLayer(nn.Module):
    def __init__(self, target_params):
        super().__init__()

        self.target_params = target_params
        self.signal_channels = None
        self.signal_index = None
        self.signal2weights = None

    def init_signal2weights(self, signal_channels, signal_index=0, groups=1):
        self.signal_channels = signal_channels
        self.signal_index = signal_index
        weight_channels = next_multiply(self.target_params, groups)
        self.signal2weights = nn.Conv2d(signal_channels, weight_channels, 1, bias=False, groups=groups)

    def forward(self, s):
        if self.signal2weights is None:
            return s

        weights = self.signal2weights(s[:, self.signal_index:self.signal_index + self.signal_channels])
        out = weights[:, :self.target_params]

        return out


class HyperPatchInvertedResidual(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, expand_ratio=1, padding_mode='reflect',
                 norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU6')):
        super().__init__()

        self.stride = stride
        assert stride in [1, 2]

        self.padding_mode = padding_mode
        self.padding = (1, 1)
        self._padding_repeated_twice = self.padding + self.padding
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.kernel_size = _pair(kernel_size)
        self.hidden_dim = int(round(in_nc * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_nc == out_nc

        self.bn1 = build_norm_layer(norm_cfg, self.hidden_dim)[1]
        self.bn2 = build_norm_layer(norm_cfg, self.hidden_dim)[1]
        self.bn3 = build_norm_layer(norm_cfg, self.out_nc)[1]
        self.act1 = build_activation_layer(act_cfg)
        self.act2 = build_activation_layer(act_cfg)

        # Calculate hyper params and weight ranges
        self.hyper_params = 0
        self._ranges = [0]
        self.hyper_params += in_nc * self.hidden_dim
        self._ranges.append(self.hyper_params)
        self.hyper_params += np.prod((self.hidden_dim,) + self.kernel_size)
        self._ranges.append(self.hyper_params)
        self.hyper_params += self.hidden_dim * out_nc
        self._ranges.append(self.hyper_params)

    def conv(self, x, weight):
        b, c, h, w = x.shape

        fh, fw = weight.shape[-2:]
        ph, pw = x.shape[-2] // fh, x.shape[-1] // fw
        kh, kw = ph + self.padding[0] * 2, pw + self.padding[1] * 2

        if self.padding_mode != 'zeros' and np.any(self._padding_repeated_twice):
            x = F.pad(x, self._padding_repeated_twice, mode=self.padding_mode)

        x = x.permute(0, 2, 3, 1).unfold(1, kh, ph).unfold(2, kw, pw).reshape(1, -1, kh, kw)

        if b == 1:
            weight = weight.permute(0, 2, 3, 1).view(-1, weight.shape[1])
        else:
            weight = weight.permute(0, 2, 3, 1).reshape(-1, weight.shape[1])

        # Conv1
        weight1 = weight[:, self._ranges[0]:self._ranges[1]]
        weight1 = weight1.reshape(b * fh * fw * self.hidden_dim, self.in_nc, 1, 1)
        x = F.conv2d(x, weight1, bias=None, groups=b * fh * fw)
        x = self.bn1(x.view(b * fh * fw, -1, kh, kw)).view(1, -1, kh, kw)
        x = self.act1(x)

        # Conv2

        weight2 = weight[:, self._ranges[1]:self._ranges[2]]
        weight2 = weight2.reshape(b * fh * fw * self.hidden_dim, 1, *self.kernel_size)
        x = F.conv2d(x, weight2, bias=None, stride=self.stride, groups=b * fh * fw * self.hidden_dim)
        x = self.bn2(x.view(b * fh * fw, -1, ph, pw)).view(1, -1, ph, pw)
        x = self.act2(x)

        # Conv3
        weight3 = weight[:, self._ranges[2]:self._ranges[3]]
        weight3 = weight3.reshape(b * fh * fw * self.out_nc, self.hidden_dim, 1, 1)
        x = F.conv2d(x, weight3, bias=None, groups=b * fh * fw)
        x = self.bn3(x.view(b * fh * fw, -1, ph, pw))

        x = x.view(b, fh, fw, -1, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, -1, h, w)

        return x

    def forward(self, x, s):
        y = self.conv(x, s)

        if self.use_res_connect:
            return x + y
        else:
            return y


class WeightMapper(nn.Module):
    """ Weight mapper module (called context head in the paper).

    Args:
        in_channels (int): input number of channels.
        out_channels (int): output number of channels.
        levels (int): number of levels operating on different strides.
        bias (bool): if True, enables bias in all convolution operations.
    """

    def __init__(self, in_channels, out_channels, levels=3, bias=False):
        super().__init__()

        assert levels > 0, 'levels must be greater than zero'
        assert in_channels % 2 == 0, 'in_channels must be divisible by 2'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.bias = bias

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        # Add blocks
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level in range(self.levels - 1):
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2, bias=bias),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True))
            )
            self.up_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 1, bias=bias),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True))
            )

    def forward(self, x):
        x = self.in_conv(x)

        # Down stream
        feat = [x]
        for level in range(self.levels - 1):
            feat.append(self.down_blocks[level](feat[-1]))

        # Average the last feature map
        orig_shape = feat[-1].shape
        if orig_shape[-2:] != (1, 1):
            x = F.adaptive_avg_pool2d(feat[-1], 1)
            x = F.interpolate(x, orig_shape[-2:], mode='nearest')

        # Up stream
        for level in range(self.levels - 2, -1, -1):
            x = torch.cat((feat.pop(-1), x), dim=1)
            x = self.up_blocks[level](x)

            x = F.interpolate(x, feat[-1].shape[-2:], mode='nearest')

        # Output head
        x = torch.cat((feat.pop(-1), x), dim=1)

        return x


def next_multiply(x, base):
    return type(x)(np.ceil(x / base) * base)


class HyperPatchNoPadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.hyper_params = np.prod((out_channels, in_channels // groups) + self.kernel_size)

    def forward(self, x, weight):
        b, c, h, w = x.shape
        fh, fw = weight.shape[-2:]
        ph, pw = x.shape[-2] // fh, x.shape[-1] // fw

        weight = weight.permute(0, 2, 3, 1)
        weight = weight.reshape(b * fh * fw * self.out_channels, self.in_channels // self.groups, *self.kernel_size)

        x = x.view(b, c, fh, ph, fw, pw).permute(0, 2, 4, 1, 3, 5).reshape(1, -1, ph, pw)
        x = F.conv2d(x, weight, bias=None, stride=self.stride, dilation=self.dilation, groups=b * fh * fw * self.groups)
        x = x.view(b, fh, fw, -1, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, -1, h, w)

        return x


class HyperPatch(nn.Module):
    padding_modes = ['zeros', 'reflect', 'replicate', 'circular']

    def __init__(self, module: nn.Module, padding=0, padding_mode='reflect'):
        super().__init__()

        if padding_mode not in self.padding_modes:
            raise ValueError(
                f"padding_mode must be one of {self.padding_modes}, but got padding_mode='{padding_mode}'")

        self.hyper_module = module
        self.padding = _pair(padding)
        self.padding_mode = padding_mode
        self._padding_repeated_twice = self.padding + self.padding

    @property
    def hyper_params(self):
        return self.hyper_module.hyper_params

    def forward(self, x, weight):
        b, c, h, w = x.shape
        fh, fw = weight.shape[-2:]
        ph, pw = x.shape[-2] // fh, x.shape[-1] // fw
        kh, kw = ph + self.padding[0] * 2, pw + self.padding[1] * 2

        weight = weight.permute(0, 2, 3, 1).reshape(-1, weight.shape[1]).contiguous()

        x = F.pad(x, self._padding_repeated_twice, mode=self.padding_mode)
        x = torch.nn.functional.unfold(x, (kh, kw), stride=(ph, pw))  # B x (C x (ph x pw)) x (fh * fw)
        x = x.transpose(1, 2).reshape(-1, c, kh, kw).contiguous()

        x = self.hyper_module(x, weight)
        x = x.view(b, fh * fw, -1, ph * pw).permute(0, 2, 3, 1).reshape(b, -1, fh * fw)
        x = F.fold(x, (h, w), kernel_size=(ph, pw), stride=(ph, pw))

        return x


class HyperPatchConv2d(HyperPatch):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='reflect', norm_weights=False):
        conv = MetaConv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups,
                          norm_weights=norm_weights)
        super().__init__(conv, padding, padding_mode)

    @property
    def in_channels(self):
        return self.hyper_module.in_channels

    @property
    def out_channels(self):
        return self.hyper_module.out_channels

    @property
    def kernel_size(self):
        return self.hyper_module.kernel_size

    @property
    def groups(self):
        return self.hyper_module.groups

    def __repr__(self):
        s = self.__class__.__name__ + '({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.hyper_module.dilation != (1,) * len(self.hyper_module.dilation):
            s += ', dilation={dilation}'
        if self.hyper_module.groups != 1:
            s += ', groups={groups}'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        s += ')'

        d = {**self.hyper_module.__dict__}
        d['padding'] = self.padding
        d['padding_mode'] = self.padding_mode

        return s.format(**d)


def make_hyper_patch_conv2d_block(in_nc, out_nc, kernel_size=3, stride=1, padding=None, dilation=1, groups=1,
                                  padding_mode='reflect', norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
    """ Defines a Hyper patch-wise convolution block with a normalization layer, an activation layer, and an optional
    dropout layer.

    Args:
        in_nc (int): Input number of channels
        out_nc (int): Output number of channels
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride
        padding (int, optional): The amount of padding for the height and width dimensions
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        norm_cfg (dict): Type of feature normalization layer
        act_cfg (dict): Type of activation layer
    """

    padding = kernel_size // 2 if padding is None else padding
    if padding == 0:
        layers = [HyperPatchNoPadding(in_nc, out_nc, kernel_size, stride, dilation, groups)]
    else:
        layers = [HyperPatchConv2d(in_nc, out_nc, kernel_size, stride, padding, dilation, groups, padding_mode)]

    if norm_cfg is not None:
        layers.append(build_norm_layer(norm_cfg, out_nc)[1])

    if act_cfg is not None:
        layers.append(build_activation_layer(act_cfg))

    return MetaSequential(*layers)


def divide_feature(in_feature, out_features, min_unit=8):
    """ Divides in_feature relative to each of the provided out_features.

    The division of the input feature will be in multiplies of "min_unit".
    The algorithm makes sure that equal output features will get the same portion of the input feature.
    The smallest out feature will receive all the round down overflow (usually the final fc)

    Args:
        in_feature: the input feature to divide
        out_features: the relative sizes of the output features
        min_unit: each division of the input feature will be divisible by this number.
        in_feature must be divisible by this number as well

    Returns:
        np.array: array of integers of the divided input feature in the size of out_features.
    """

    assert in_feature % min_unit == 0, f'in_feature ({in_feature}) must be divisible by min_unit ({min_unit})'

    units = in_feature // min_unit
    indices = np.argsort(out_features)
    out_features_sorted = np.array(out_features)[indices]
    out_feat_groups = [(k, indices[list(g)]) for k, g in groupby(range(len(indices)), lambda i: out_features_sorted[i])]
    out_feat_groups.sort(key=lambda x: x[0] * len(x[1]), reverse=True)
    units_feat_ratio = float(units) / sum(out_features)

    # For each feature group
    out_group_units = [len(out_feat_group[1]) for out_feat_group in out_feat_groups]
    remaining_units = units - sum(out_group_units)
    for i, out_feat_group in enumerate(out_feat_groups):    # out_feat_group: (out_feature, indices array)
        if i < (len(out_feat_groups) - 1):
            n = len(out_feat_group[1])  # group size
            curr_out_feat_size = out_feat_group[0] * n
            curr_units = max(curr_out_feat_size * units_feat_ratio, n)
            curr_units = curr_units // n * n - n  # Make divisible by num elements
            curr_units = min(curr_units, remaining_units)
            out_group_units[i] += curr_units
            remaining_units -= curr_units
            if remaining_units == 0:
                break
        else:
            out_group_units[-1] += remaining_units

    # Final feature division
    divided_in_features = np.zeros(len(out_features), dtype=int)
    for i, out_feat_group in enumerate(out_feat_groups):
        for j in range(len(out_feat_group[1])):
            divided_in_features[out_feat_group[1][j]] = out_group_units[i] // len(out_feat_group[1]) * min_unit

    return divided_in_features


@HEADS.register_module()
class HyperSegHead(BaseDecodeHead):
    """ Hypernetwork generator comprised of a backbone network, weight mapper, and a decoder.

    Args:
        backbone (nn.Module factory): Backbone network
        weight_mapper (nn.Module factory): Weight mapper network.
        in_nc (int): input number of channels.
        num_classes (int): output number of classes.
        kernel_sizes (int): the kernel size of the decoder layers.
        level_layers (int): number of layers in each level of the decoder.
        level_channels (list of int, optional): If specified, sets the output channels of each level in the decoder.
        expand_ratio (int): inverted residual block's expansion ratio in the decoder.
        weight_groups (int, optional): per level signal to weights groups in the decoder.
        with_out_fc (bool): If True, add a final fully connected layer to the decoder.
        decoder_groups (int, optional): per level groups in the decoder.
        decoder_dropout (float): If specified, enables dropout with the given probability.
        coords_res (list of tuple of int, optional): list of inference resolutions for caching positional embedding.
        unify_level (int, optional): the starting level to unify the signal to weights operation from.
    """

    def __init__(self, kernel_sizes=3, level_layers=1,
                 level_channels=None, expand_ratio=1, weight_groups=1,
                 with_out_fc=False, with_out_norm=False, decoder_groups=1, decoder_dropout=None,
                 coords_res=None, unify_level=None, weight_levels=3, weight_same_last_level=False, **kwargs):
        super().__init__(input_transform='multiple_select', enable_out_seg=False, **kwargs)

        self.weight_same_last_level = weight_same_last_level

        feat_channels = self.in_channels if self.weight_same_last_level else self.in_channels[:-1]
        self.decoder = MultiScaleDecoder(
            feat_channels, self.in_channels[-1], self.num_classes,
            kernel_sizes, level_layers, level_channels,
            with_out_fc=with_out_fc, with_out_norm=with_out_norm, out_kernel_size=1,
            expand_ratio=expand_ratio, groups=decoder_groups,
            weight_groups=weight_groups, dropout=decoder_dropout,
            coords_res=coords_res, unify_level=unify_level,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )
        self.weight_mapper = WeightMapper(
            self.in_channels[-1], self.decoder.param_groups,
            levels=weight_levels
        )

    def forward(self, inputs):
        all_features = self._transform_inputs(inputs)

        if self.weight_same_last_level:
            main_features, weight_features = all_features, all_features[-1]
        else:
            main_features, weight_features = all_features[:-1], all_features[-1]

        dynamic_weights = self.weight_mapper(weight_features)
        out = self.decoder(main_features, dynamic_weights)

        return out
