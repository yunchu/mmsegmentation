# Copyright (c) 2018-2019 Maxim Berman
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Modified from https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytor
ch/lovasz_losses.py Lovasz-Softmax and Jaccard hinge loss in PyTorch Maxim
Berman 2018 ESAT-PSI KU Leuven (MIT License)"""

import mmcv
import torch
import torch.nn.functional as F

from ..builder import LOSSES, build_scheduler
from .utils import get_class_weight, weight_reduce_loss
from .base import BaseWeightedLoss


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors.

    See Alg. 1 in paper.
    """

    gt_sum = gt_sorted.sum()
    intersection = gt_sum - gt_sorted.float().cumsum(0)
    union = gt_sum + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union

    p = len(gt_sorted)
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

    return jaccard


def flatten_binary_logits(logits, labels, ignore_index=None):
    """Flattens predictions in the batch (binary case) Remove labels equal to
    'ignore_index'."""

    logits = logits.view(-1)
    labels = labels.view(-1)
    if ignore_index is None:
        return logits, labels

    valid = labels != ignore_index
    valid_logits = logits[valid]
    valid_labels = labels[valid]

    return valid_logits, valid_labels


def flatten_probs(probs, labels, ignore_index=None):
    """Flattens predictions in the batch."""

    if probs.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probs.size()
        probs = probs.view(B, 1, H, W)
    B, C, H, W = probs.size()

    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B*H*W, C=P,C
    labels = labels.view(-1)
    if ignore_index is None:
        return probs, labels

    valid = labels != ignore_index
    valid_probs = probs[valid]
    valid_labels = labels[valid]

    return valid_probs, valid_labels


def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss.

    Args:
        logits (torch.Tensor): [P], logits at each prediction
            (between -infty and +infty).
        labels (torch.Tensor): [P], binary ground truth labels (0 or 1).

    Returns:
        torch.Tensor: The calculated loss.
    """

    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.

    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)

    return loss


def lovasz_hinge(logits,
                 labels,
                 classes='present',
                 per_image=False,
                 class_weight=None,
                 reduction='mean',
                 avg_factor=None,
                 ignore_index=255):
    """Binary Lovasz hinge loss.

    Args:
        logits (torch.Tensor): [B, H, W], logits at each pixel
            (between -infty and +infty).
        labels (torch.Tensor): [B, H, W], binary ground truth masks (0 or 1).
        classes (str | list[int], optional): Placeholder, to be consistent with
            other loss. Default: None.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
        class_weight (list[float], optional): Placeholder, to be consistent
            with other loss. Default: None.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. This parameter only works when per_image is True.
            Default: None.
        ignore_index (int | None): The label index to be ignored. Default: 255.

    Returns:
        torch.Tensor: The calculated loss.
    """

    if per_image:
        loss = [
            lovasz_hinge_flat(*flatten_binary_logits(logit.unsqueeze(0), label.unsqueeze(0), ignore_index))
            for logit, label in zip(logits, labels)
        ]
        loss = weight_reduce_loss(torch.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_logits(logits, labels, ignore_index))

    return loss


def lovasz_softmax_flat(probs, labels, classes='present', class_weight=None):
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (torch.Tensor): [P, C], class probabilities at each prediction
            (between 0 and 1).
        labels (torch.Tensor): [P], ground truth labels (between 0 and C - 1).
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        class_weight (list[float], optional): The weight for each class.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss.
    """

    if probs.numel() == 0:
        # only void pixels, the gradients should be 0
        return probs * 0.

    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue

        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]

        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        loss = torch.dot(errors_sorted, lovasz_grad(fg_sorted))

        if class_weight is not None:
            loss *= class_weight[c]

        losses.append(loss)

    return torch.stack(losses).mean()


def lovasz_softmax(probs,
                   labels,
                   classes='present',
                   per_image=False,
                   class_weight=None,
                   reduction='mean',
                   avg_factor=None,
                   ignore_index=255):
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (torch.Tensor): [B, C, H, W], class probabilities at each
            prediction (between 0 and 1).
        labels (torch.Tensor): [B, H, W], ground truth labels (between 0 and
            C - 1).
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. This parameter only works when per_image is True.
            Default: None.
        ignore_index (int | None): The label index to be ignored. Default: 255.

    Returns:
        torch.Tensor: The calculated loss.
    """

    if per_image:
        loss = [
            lovasz_softmax_flat(
                *flatten_probs(prob.unsqueeze(0), label.unsqueeze(0), ignore_index),
                classes=classes,
                class_weight=class_weight)
            for prob, label in zip(probs, labels)
        ]
        loss = weight_reduce_loss(torch.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_softmax_flat(
            *flatten_probs(probs, labels, ignore_index),
            classes=classes,
            class_weight=class_weight
        )

    return loss


@LOSSES.register_module()
class LovaszLoss(BaseWeightedLoss):
    """LovaszLoss.

    This loss is proposed in `The Lovasz-Softmax loss: A tractable surrogate
    for the optimization of the intersection-over-union measure in neural
    networks <https://arxiv.org/abs/1705.08790>`_.

    Args:
        loss_type (str, optional): Binary or multi-class loss.
            Default: 'multi_class'. Options are "binary" and "multi_class".
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 loss_type='multi_class',
                 classes='present',
                 per_image=False,
                 class_weight=None,
                 scale_cfg=None,
                 **kwargs):
        super(LovaszLoss, self).__init__(**kwargs)

        assert loss_type in ('binary', 'multi_class'), "loss_type should be 'binary' or 'multi_class'."
        assert classes in ('all', 'present') or mmcv.is_list_of(classes, int)
        if not per_image:
            assert self.reduction == 'none', "reduction should be 'none' when per_image is False."

        if loss_type == 'binary':
            self.cls_criterion = lovasz_hinge
        else:
            self.cls_criterion = lovasz_softmax

        self.classes = classes
        self.per_image = per_image
        self.class_weight = get_class_weight(class_weight)

        self.last_scale, self.scale_scheduler = None, None
        if scale_cfg is not None:
            self.scale_scheduler = build_scheduler(scale_cfg)
        if self.cls_criterion == lovasz_softmax:
            assert self.scale_scheduler is not None

    @property
    def name(self):
        return 'lovasz'

    def _forward(self,
                 cls_score,
                 label,
                 pixel_weights=None,
                 avg_factor=None,
                 reduction_override=None):
        """Forward function."""

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        # if multi-class loss, transform logits to probs
        if self.cls_criterion == lovasz_softmax:
            self.last_scale = self.scale_scheduler.get_scale_and_increment_step()
            cls_score = F.softmax(self.last_scale * cls_score, dim=1)

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            self.classes,
            self.per_image,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index
        )

        return loss_cls
