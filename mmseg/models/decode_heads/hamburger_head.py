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
    def __init__(self, S=1, D=512, R=64, spatial=True, train_steps=6, eval_steps=6,
                 inv_t=100, eta=0.9, rand_init=True, **kwargs):
        super().__init__()

        self.S = S
        self.D = D
        self.R = R
        self.spatial = spatial
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.inv_t = inv_t
        self.eta = eta
        self.rand_init = rand_init

        self.bases = None

    @abstractmethod
    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    @abstractmethod
    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @abstractmethod
    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        if not self.rand_init and not self.training and not return_bases:
            self.online_update(bases)

        return x

    @torch.no_grad()
    def online_update(self, bases):
        # (B, S, D, R) -> (S, D, R)
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)


class VQ2D(_MatrixDecomposition2DBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.randn((B * S, D, R)).cuda()
        else:
            bases = torch.randn((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, bases, _):
        # (B * S, D, N), normalize x along D (for cosine similarity)
        std_x = F.normalize(x, dim=1)

        # (B * S, D, R), normalize bases along D (for cosine similarity)
        std_bases = F.normalize(bases, dim=1, eps=1e-6)

        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(std_x.transpose(1, 2), std_bases)

        # softmax along R
        coef = F.softmax(self.inv_t * coef, dim=-1)

        # normalize along N
        coef = coef / (1e-6 + coef.sum(dim=1, keepdim=True))

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        bases = torch.bmm(x, coef)

        return bases, coef

    def compute_coef(self, x, bases, _):
        with torch.no_grad():
            # (B * S, D, N) -> (B * S, 1, N)
            x_norm = x.norm(dim=1, keepdim=True)

        # (B * S, D, N) / (B * S, 1, N) -> (B * S, D, N)
        std_x = x / (1e-6 + x_norm)

        # (B * S, D, R), normalize bases along D (for cosine similarity)
        std_bases = F.normalize(bases, dim=1, eps=1e-6)

        # (B * S, N, D)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(std_x.transpose(1, 2), std_bases)

        # softmax along R
        coef = F.softmax(self.inv_t * coef, dim=-1)

        return coef


class CD2D(_MatrixDecomposition2DBase):
    def __init__(self, beta=0.1, **kwargs):
        super().__init__(**kwargs)

        self.beta = beta

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.randn((B * S, D, R)).cuda()
        else:
            bases = torch.randn((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, bases, _):
        # normalize x along D (for cosine similarity)
        std_x = F.normalize(x, dim=1)

        # (B * S, N, D) @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(std_x.transpose(1, 2), bases)

        # softmax along R
        coef = F.softmax(self.inv_t * coef, dim=-1)

        # normalize along N
        coef = coef / (1e-6 + coef.sum(dim=1, keepdim=True))

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        bases = torch.bmm(x, coef)

        # normalize along D
        bases = F.normalize(bases, dim=1, eps=1e-6)

        return bases, coef

    def compute_coef(self, x, bases, _):
        # [(B * S, R, D) @ (B * S, D, R) + (B * S, R, R)] ^ (-1) -> (B * S, R, R)
        temp = torch.bmm(bases.transpose(1, 2), bases) \
            + self.beta * torch.eye(self.R).repeat(x.shape[0], 1, 1).cuda()
        temp = torch.inverse(temp)

        # (B * S, D, N)^T @ (B * S, D, R) @ (B * S, R, R) -> (B * S, N, R)
        coef = x.transpose(1, 2).bmm(bases).bmm(temp)

        return coef


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


def build_ham(key, **kwargs):
    hams = {'VQ': VQ2D,
            'CD': CD2D,
            'NMF': NMF2D}
    assert key in hams

    return hams[key](**kwargs)


class HamburgerV2Plus(nn.Module):
    def __init__(self, in_c, S=1, D=512, R=64, ham_type='NMF', dual=False, zero_ham=True, cheese_factor=1,
                 conv_cfg=None, norm_cfg=None, act_cfg=dict(type='ReLU')):
        super().__init__()

        S = S
        D = D
        C = S * D

        self.dual = dual
        if self.dual:
            C = 2 * C

        if ham_type == 'NMF':
            self.lower_bread = nn.Sequential(nn.Conv2d(in_c, C, 1),
                                             nn.ReLU(inplace=True))
        else:
            self.lower_bread = nn.Conv2d(in_c, C, 1)

        if self.dual:
            self.ham_1 = build_ham(ham_type, S=S, D=D, R=R, spatial=True)
            self.ham_2 = build_ham(ham_type, S=S, D=D, R=R, spatial=False)
        else:
            self.ham = build_ham(ham_type, S=S, D=D, R=R, spatial=True)

        if self.dual:
            cheese_factor = 2 * cheese_factor

        self.cheese = ConvModule(C, C // cheese_factor, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.upper_bread = nn.Conv2d(C // cheese_factor, in_c, 1, bias=False)

        self.coef_shortcut = nn.Parameter(torch.tensor([1.0]))
        self.coef_ham = nn.Parameter(torch.tensor([0.0 if zero_ham else 1.0]))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / N) ** 0.5)
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        shortcut = x

        x = self.lower_bread(x)

        if self.dual:
            x = x.view(x.shape[0], 2, x.shape[1] // 2, *x.shape[2:])
            x_1 = self.ham_1(x.narrow(1, 0, 1).squeeze(dim=1))
            x_2 = self.ham_2(x.narrow(1, 1, 1).squeeze(dim=1))
            x = torch.cat([x_1, x_2], dim=1)
        else:
            x = self.ham(x)
        x = self.cheese(x)
        x = self.upper_bread(x)

        x = self.coef_ham * x + self.coef_shortcut * shortcut
        x = F.relu(x, inplace=True)

        return x

    def online_update(self, bases):
        if hasattr(self.ham, 'online_update'):
            self.ham.online_update(bases)


@HEADS.register_module()
class HamburgerHead(BaseDecodeHead):
    """HamNet.

    This head is implemented of `HamNet <https://arxiv.org/abs/2109.04553>`_.

    """

    def __init__(self, reduced_channels=8, **kwargs):
        super().__init__(**kwargs)

        self.reduced_channels = reduced_channels

        self.hamburger = HamburgerV2Plus(
            self.in_channels,
            D=self.channels,
            R=self.reduced_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.conv = ConvModule(
            self.channels,
            self.channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def forward(self, inputs):
        """Forward function."""

        x = self._transform_inputs(inputs)

        y = self.hamburger(x)
        y = self.conv(y)

        logits = self.cls_seg(y)

        return logits
