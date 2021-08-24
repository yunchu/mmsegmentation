# The original repo: https://github.com/bes-dev/mpl.pytorch

import torch

from ..builder import PIXEL_SAMPLERS
from .base_pixel_sampler import BasePixelSampler

from ....utils.ext_loader import load_ext
ext_module = load_ext('_mpl', ['compute_weights'])


@PIXEL_SAMPLERS.register_module()
class MaxPoolingPixelSampler(BasePixelSampler):
    """Max-Pooling Loss
    Implementation of "Loss Max-Pooling for Semantic Image Segmentation"
    https://arxiv.org/abs/1704.02966
    """

    def __init__(self, context, ratio=0.3, p=1.7):
        super().__init__(context)

        assert 0 < ratio <= 1, "ratio should be in range [0, 1]"
        assert p > 1, "p should be > 1"

        self.ratio = ratio
        self.p = p

    def sample(self, seg_logit, seg_label):
        with torch.no_grad():
            assert seg_logit.shape[2:] == seg_label.shape[2:]
            assert seg_label.shape[1] == 1

            seg_label = seg_label.squeeze(1).long()
            valid_mask = seg_label != self.context.ignore_index
            seg_weight = seg_logit.new_zeros(size=seg_label.size())

            main_loss_idx = self.context.sampler_loss_idx
            losses = self.context.loss_modules[main_loss_idx](
                seg_logit,
                seg_label,
                weight=None,
                ignore_index=self.context.ignore_index,
                reduction_override='none',
                increment_train_step=False,
            )

            losses = losses.view(-1)
            sort_losses, sort_indices = losses[valid_mask.view(-1)].sort()

            weights = torch.zeros(sort_losses.size())
            ext_module.compute_weights(
                sort_losses.size(0),
                sort_losses.cpu(),
                sort_indices.cpu(),
                weights.cpu(),
                self.ratio,
                self.p
            )

            scale = float(sort_losses.size(0))
            seg_weight[valid_mask] = scale * weights.to(seg_logit.device)

            return seg_weight
