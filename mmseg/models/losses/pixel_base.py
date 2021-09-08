from abc import abstractmethod

import torch
import torch.nn.functional as F

from mmseg.core import entropy
from .base import BaseWeightedLoss
from .utils import weight_reduce_loss
from .. import builder


class BasePixelLoss(BaseWeightedLoss):
    def __init__(self,
                 scale_cfg=None,
                 pr_product=False,
                 conf_penalty_weight=None,
                 border_reweighting=False,
                 **kwargs):
        super(BasePixelLoss, self).__init__(**kwargs)

        self._enable_pr_product = pr_product
        self._border_reweighting = border_reweighting

        self._reg_weight_scheduler = builder.build_scheduler(conf_penalty_weight)
        self._scale_scheduler = builder.build_scheduler(scale_cfg, default_value=1.0)

        self._last_scale = 0.0
        self._last_reg_weight = 0.0

    @property
    def last_scale(self):
        return self._last_scale

    @property
    def last_reg_weight(self):
        return self._last_reg_weight

    @property
    def with_regularization(self):
        return self._reg_weight_scheduler is not None

    @property
    def with_pr_product(self):
        return self._enable_pr_product

    @property
    def with_border_reweighting(self):
        return self._border_reweighting

    @staticmethod
    def _pr_product(prod):
        alpha = torch.sqrt(1.0 - prod.pow(2.0))
        out_prod = alpha.detach() * prod + prod.detach() * (1.0 - alpha)

        return out_prod

    @staticmethod
    def _regularization(logits, scale, weight):
        probs = F.softmax(scale * logits, dim=1)
        entropy_values = entropy(probs, dim=1)
        out_values = -weight * entropy_values

        return out_values

    @staticmethod
    def _sparsity(values, valid_mask):
        with torch.no_grad():
            valid_values = values[valid_mask]
            sparsity = 1.0 - float(valid_values.count_nonzero().item()) / max(1.0, float(valid_mask.sum().item()))

            return sparsity

    def _forward(self, output, labels, avg_factor=None, pixel_weights=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        self._last_scale = self._scale_scheduler(self.iter)

        if self.with_pr_product:
            output = self._pr_product(output)

        num_classes = output.size(1)
        valid_labels = torch.clamp(labels, 0, num_classes - 1)
        valid_mask = labels != self.ignore_index

        losses, updated_output = self._calculate(output, valid_labels, self._last_scale)

        if self.with_regularization:
            self._last_reg_weight = self._reg_weight_scheduler(self.iter)
            regularization = self._regularization(updated_output, self._last_scale, self._last_reg_weight)
            losses = torch.clamp_min(losses + regularization, 0.0)

        if self.with_border_reweighting:
            assert pixel_weights is not None
            losses = pixel_weights.squeeze(1) * losses

        losses = torch.where(valid_mask, losses, torch.zeros_like(losses))
        raw_sparsity = self._sparsity(losses, valid_mask)

        weight, weight_sparsity = None, 0.0
        if self.sampler is not None:
            weight = self.sampler(losses, output, valid_labels, valid_mask)
            weight_sparsity = self._sparsity(weight, valid_mask)

        loss = weight_reduce_loss(
            losses,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor
        )

        meta = dict(
            weight=self.last_loss_weight,
            reg_weight=self.last_reg_weight,
            scale=self.last_scale,
            raw_sparsity=raw_sparsity,
            weight_sparsity=weight_sparsity
        )

        return loss, meta

    @abstractmethod
    def _calculate(self, output, labels, scale):
        pass
