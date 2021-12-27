# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F
import numpy as np


def entropy(p, dim=1, keepdim=False):
    return -torch.where(p > 0.0, p * p.log(), torch.zeros_like(p)).sum(dim=dim, keepdim=keepdim)


def focal_loss(input_values, gamma):
    return (1.0 - torch.exp(-input_values)) ** gamma * input_values


class MaxEntropyLoss:
    def __init__(self, scale=1.0):
        super(MaxEntropyLoss, self).__init__()

        self.scale = scale
        assert self.scale > 0.0

    def forward(self, cos_theta):
        probs = F.softmax(self.scale * cos_theta, dim=1)

        entropy_values = entropy(probs, dim=1)
        losses = np.log(cos_theta.size(-1)) - entropy_values

        return losses.mean()


class CrossEntropy:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, logits, target):
        return self.weight * F.cross_entropy(logits, target, reduction='none')


class CrossEntropySmooth:
    def __init__(self, epsilon=0.1, weight=1.0):
        self.epsilon = epsilon
        self.weight = weight

    def __call__(self, logits, target):
        with torch.no_grad():
            b, n, h, w = logits.size()
            assert n > 1

            target_value = 1.0 - self.epsilon
            blank_value = self.epsilon / float(n - 1)

            targets = logits.new_full((b * h * w, n), blank_value).scatter_(1, target.view(-1, 1), target_value)
            targets = targets.view(b, h, w, n).permute(0, 3, 1, 2)

        log_softmax = F.log_softmax(logits, dim=1)
        losses = torch.neg((targets * log_softmax).sum(dim=1))

        return self.weight * losses


class NormalizedCrossEntropy:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, logits, target):
        log_softmax = F.log_softmax(logits, dim=1)
        b, c, h, w = log_softmax.size()

        log_softmax = log_softmax.permute(0, 2, 3, 1).reshape(-1, c)
        target = target.view(-1)

        target_log_softmax = log_softmax[torch.arange(target.size(0), device=target.device), target]
        target_log_softmax = target_log_softmax.view(b, h, w)

        sum_log_softmax = log_softmax.sum(dim=1)
        losses = self.weight * target_log_softmax / sum_log_softmax

        return losses


class ReverseCrossEntropy:
    def __init__(self, scale=4.0, weight=1.0):
        self.weight = weight * abs(float(scale))

    def __call__(self, logits, target):
        all_probs = F.softmax(logits, dim=1)
        b, c, h, w = all_probs.size()

        all_probs = all_probs.permute(0, 2, 3, 1).reshape(-1, c)
        target = target.view(-1)

        target_probs = all_probs[torch.arange(target.size(0), device=target.device), target]
        target_probs = target_probs.view(b, h, w)

        losses = self.weight * (1.0 - target_probs)

        return losses


class SymmetricCrossEntropy:
    def __init__(self, alpha=1.0, beta=1.0):
        self.ce = CrossEntropy(weight=alpha)
        self.rce = ReverseCrossEntropy(weight=beta)

    def __call__(self, logits, target):
        return self.ce(logits, target) + self.rce(logits, target)


class ActivePassiveLoss:
    def __init__(self, alpha=100.0, beta=1.0):
        self.active_loss = NormalizedCrossEntropy(weight=alpha)
        self.passive_loss = ReverseCrossEntropy(weight=beta)

    def __call__(self, logits, target):
        return self.active_loss(logits, target) + self.passive_loss(logits, target)


def build_classification_loss(name, **kwargs):
    if name == 'ce':
        return CrossEntropy(**kwargs)
    elif name == 'ce_smooth':
        return CrossEntropySmooth(**kwargs)
    elif name == 'nce':
        return NormalizedCrossEntropy(**kwargs)
    elif name == 'rce':
        return ReverseCrossEntropy(**kwargs)
    elif name == 'sl':
        return SymmetricCrossEntropy(**kwargs)
    elif name == 'apl':
        return ActivePassiveLoss(**kwargs)
    else:
        raise AttributeError('Unknown name of loss: {}'.format(name))
