# Copyright (C) 2018-2021 kornia
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Modified from https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/tversky.html"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight


def tversky_loss(pred,
                 target,
                 valid_mask,
                 alpha,
                 beta,
                 eps=1e-6,
                 class_weight=None,
                 reduction='mean',
                 avg_factor=None,
                 ignore_index=255,
                 **kwargs):
    assert pred.shape[0] == target.shape[0]

    num_classes = pred.shape[1]
    if num_classes == 1:
        class_ids = [0] if ignore_index != 0 else []
    elif num_classes == 2:
        class_ids = [1] if ignore_index != 1 else []
    else:
        class_ids = [i for i in range(num_classes) if i != ignore_index]
    assert len(class_ids) >= 1

    class_losses = []
    for i in class_ids:
        tversky_loss_value = binary_tversky_loss(
            pred[:, i],
            target[..., i],
            valid_mask=valid_mask,
            alpha=alpha,
            beta=beta,
            eps=eps,
        )

        if class_weight is not None:
            tversky_loss_value *= class_weight[i]

        class_losses.append(tversky_loss_value)

    if avg_factor is None:
        if reduction == 'mean':
            loss = sum(class_losses) / float(len(class_losses))
        elif reduction == 'sum':
            loss = sum(class_losses)
        elif reduction == 'none':
            loss = class_losses
        else:
            raise ValueError(f'unknown reduction type: {reduction}')
    else:
        if reduction == 'mean':
            loss = sum(class_losses) / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss


def binary_tversky_loss(pred, target, valid_mask, alpha, beta, eps=1e-6):
    assert pred.shape[0] == target.shape[0]

    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    valid_pred = torch.mul(pred, valid_mask)
    valid_target = torch.mul(target, valid_mask)

    intersection = torch.sum(valid_pred * valid_target, dim=1)
    fps = torch.sum(valid_pred * (1.0 - valid_target), dim=1)
    fns = torch.sum((1.0 - valid_pred) * valid_target, dim=1)

    numerator = intersection
    denominator = intersection + alpha * fps + beta * fns

    return 1.0 - numerator / (denominator + eps)


@LOSSES.register_module()
class TverskyLoss(nn.Module):
    """TverskyLoss.

    This loss is proposed in `Tversky loss function for image segmentation
    using 3D fully convolutional deep networks <https://arxiv.org/abs/1706.05721>`_.
    """

    def __init__(self,
                 alpha=0.3,
                 beta=0.7,
                 eps=1e-6,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(TverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight

    @property
    def name(self):
        return 'tversky'

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes
        )

        valid_mask = (target != ignore_index).long()

        loss = self.loss_weight * tversky_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            alpha=self.alpha,
            beta=self.beta,
            eps=self.eps,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            **kwargs
        )

        return loss
