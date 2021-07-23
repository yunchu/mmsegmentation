"""Modified from https://github.com/LIVIAETS/boundary-loss"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss


def boundary_loss(pred,
                  target,
                  dist,
                  valid_mask,
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
        loss_values = binary_boundary_loss(
            pred[:, i], target[..., i], dist[..., i], valid_mask
        )

        if class_weight is not None:
            loss_values = class_weight[i] * loss_values

        class_losses.append(loss_values)

    loss = sum(class_losses)
    loss = weight_reduce_loss(
        loss,
        reduction=reduction,
        avg_factor=avg_factor
    )

    return loss


def binary_boundary_loss(pred, target, dist, valid_mask):
    assert pred.shape[0] == target.shape[0]

    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    dist = dist.reshape(dist.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    valid_pred = torch.mul(pred, valid_mask)
    valid_target = torch.mul(target, valid_mask)

    fps = valid_pred * (1.0 - valid_target)
    fns = (1.0 - valid_pred) * valid_target

    loss = (fps + fns) * dist

    return loss


def one_hot2dist(target):
    b, h, w, c = target.size()

    target = target.permute(0, 3, 1, 2).contiguous().view(-1, h, w)
    num_slices = target.size(0)

    target_cpu = target.cpu().numpy()
    dist_cpu = np.zeros_like(target_cpu, dtype=np.float32)

    for slice_id in range(num_slices):
        pos_mask = target_cpu[slice_id].astype(np.bool)

        if pos_mask.any():
            neg_mask = ~pos_mask

            fp_dist = distance_transform_edt(neg_mask) * neg_mask
            fn_dist = distance_transform_edt(pos_mask) * pos_mask
            dist_cpu[slice_id] = fp_dist + fn_dist

    dist = torch.from_numpy(dist_cpu).to(target.device)
    out = dist.view(b, c, h, w).permute(0, 2, 3, 1)

    return out


@LOSSES.register_module()
class BoundaryLoss(nn.Module):
    """BoundaryLoss.

    This loss is proposed in `Boundary loss for highly unbalanced
    segmentation <https://arxiv.org/abs/1812.07032>`.
    """

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(BoundaryLoss, self).__init__()

        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight

    @property
    def name(self):
        return 'boundary'

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        class_weight = None
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes
        )

        dist = one_hot2dist(one_hot_target)
        valid_mask = (target != ignore_index).long()

        loss = self.loss_weight * boundary_loss(
            pred,
            one_hot_target,
            dist,
            valid_mask=valid_mask,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            **kwargs
        )

        return loss
