import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import DepthwiseSeparableConvModule


class IterativeAggregator(nn.Module):
    """The original repo: https://github.com/HRNet/Lite-HRNet"""

    def __init__(self, in_channels, conv_cfg=None, norm_cfg=dict(type='BN')):
        super().__init__()

        num_branches = len(in_channels)
        self.in_channels = in_channels[::-1]

        projects = []
        for i in range(num_branches):
            if i != num_branches - 1:
                out_channels = self.in_channels[i + 1]
            else:
                out_channels = self.in_channels[i]

            projects.append(DepthwiseSeparableConvModule(
                in_channels=self.in_channels[i],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'),
                dw_act_cfg=None,
                pw_act_cfg=dict(type='ReLU')
            ))

        self.projects = nn.ModuleList(projects)

    def forward(self, x):
        x = x[::-1]

        y_list = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(
                    last_x,
                    size=s.size()[-2:],
                    mode='bilinear',
                    align_corners=True
                )
                s = s + last_x

            s = self.projects[i](s)
            last_x = s

            y_list.append(s)

        return y_list[::-1]
