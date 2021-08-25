from abc import abstractmethod

import torch
import torch.nn.functional as F
from scipy.special import erfinv

from mmseg.core import entropy
from .base import BaseWeightedLoss
from .. import builder
from .utils import weight_reduce_loss


class BasePixelLoss(BaseWeightedLoss):
    def __init__(self, scale_cfg, pr_product=False, conf_penalty_weight=None, reduction='mean',
                 loss_jitter_prob=None, loss_jitter_momentum=0.1, **kwargs):
        super(BasePixelLoss, self).__init__(**kwargs)

        self._enable_pr_product = pr_product
        self._conf_penalty_weight = conf_penalty_weight
        self._reduction = reduction

        self._smooth_loss = None
        self._jitter_sigma_factor = None
        self._loss_jitter_momentum = loss_jitter_momentum
        assert 0.0 < self._loss_jitter_momentum < 1.0
        if loss_jitter_prob is not None:
            assert 0.0 < loss_jitter_prob < 1.0
            self._jitter_sigma_factor = 1.0 / ((2.0 ** 0.5) * erfinv(1.0 - 2.0 * loss_jitter_prob))

        self._scale_scheduler = builder.build_scheduler(scale_cfg)
        self._last_scale = None

    @property
    def last_scale(self):
        return self._last_scale

    @property
    def with_regularization(self):
        return self._conf_penalty_weight is not None and self._conf_penalty_weight > 0.0

    @property
    def with_pr_product(self):
        return self._enable_pr_product

    @property
    def with_loss_jitter(self):
        return self._jitter_sigma_factor is not None

    @staticmethod
    def _pr_product(prod):
        alpha = torch.sqrt(1.0 - prod.pow(2.0))
        out_prod = alpha.detach() * prod + prod.detach() * (1.0 - alpha)

        return out_prod

    def _regularization(self, logits, scale):
        probs = F.softmax(scale * logits, dim=1)
        entropy_values = entropy(probs, dim=1)
        out_values = -self._conf_penalty_weight * entropy_values

        return out_values

    def _forward(self, output, labels, avg_factor=None,
                 reduction_override=None, increment_train_step=True):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self._reduction)

        if increment_train_step:
            self._last_scale = self._scale_scheduler.get_scale_and_increment_step()
        else:
            self._last_scale = self._scale_scheduler.get_scale()

        if self.with_pr_product:
            output = self._pr_product(output)

        num_classes = output.size(1)
        valid_labels = torch.clamp(labels, 0, num_classes - 1)
        losses = self._calculate(output, valid_labels, self._last_scale)

        if self.with_regularization:
            regularization = self._regularization(output, self._last_scale)
            losses = losses + regularization

        valid_mask = labels != self.ignore_index
        losses = torch.where(valid_mask, losses, torch.zeros_like(losses))

        pixel_weights = None
        if self.sampler is not None:
            pixel_weights = self.sampler(losses, valid_mask)

        loss = weight_reduce_loss(
            losses,
            weight=pixel_weights,
            reduction=reduction,
            avg_factor=avg_factor
        )

        if self.with_loss_jitter and loss.numel() == 1:
            if self._smooth_loss is None:
                self._smooth_loss = loss.item()
            else:
                self._smooth_loss = (1.0 - self._loss_jitter_momentum) * self._smooth_loss + \
                                    self._loss_jitter_momentum * loss.item()

            jitter_sigma = self._jitter_sigma_factor * abs(self._smooth_loss)
            jitter_point = torch.normal(0.0, jitter_sigma, [], device=loss.device, dtype=loss.dtype)
            loss = (loss - jitter_point).abs() + jitter_point

        return loss

    @abstractmethod
    def _calculate(self, output, labels, scale):
        pass
