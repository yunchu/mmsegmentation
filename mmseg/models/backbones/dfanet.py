# The original repo: https://github.com/jandylin/DFANet_PyTorch

import math

import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.utils import get_root_logger
from .xception_a import XceptionA as backbone
from ..builder import BACKBONES


@BACKBONES.register_module()
class DFANet(nn.Module):
    def __init__(self,
                 extra,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False):
        super(DFANet, self).__init__()

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        self.extra = extra
        self.fc_channels = self.extra['fc_channels']  # should be 1000
        backbone_args = dict(fc_channels=self.fc_channels)

        self.backbone1 = backbone(backbone_args, conv_cfg=conv_cfg, norm_cfg=norm_cfg, norm_eval=norm_eval)
        self.backbone2 = backbone(backbone_args, conv_cfg=conv_cfg, norm_cfg=norm_cfg, norm_eval=norm_eval)
        self.backbone3 = backbone(backbone_args, conv_cfg=conv_cfg, norm_cfg=norm_cfg, norm_eval=norm_eval)

        self.backbone1_up = nn.UpsamplingBilinear2d(scale_factor=4)
        self.backbone2_up = nn.UpsamplingBilinear2d(scale_factor=4)

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
        enc1_2, enc1_3, enc1_4, fca1 = self.backbone1(x)
        fca1_up = self.backbone1_up(fca1)

        enc2_2, enc2_3, enc2_4, fca2 = self.backbone2.forward_concat(fca1_up, enc1_2, enc1_3, enc1_4)
        fca2_up = self.backbone2_up(fca2)

        enc3_2, enc3_3, enc3_4, fca3 = self.backbone3.forward_concat(fca2_up, enc2_2, enc2_3, enc2_4)

        y_list = [enc1_2, enc2_2, enc3_2, fca1, fca2, fca3]

        return y_list

    def train(self, mode=True):
        """Convert the model into training mode."""

        super().train(mode)

        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
