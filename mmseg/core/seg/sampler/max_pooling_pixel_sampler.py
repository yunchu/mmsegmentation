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

    def __init__(self, ratio=0.3, p=1.7, **kwargs):
        super().__init__(**kwargs)

        assert 0 < ratio <= 1, "ratio should be in range [0, 1]"
        assert p > 1, "p should be > 1"

        self.ratio = ratio
        self.p = p

    def _sample(self, losses=None, seg_logit=None, seg_label=None, valid_mask=None):
        assert losses is not None

        with torch.no_grad():
            if valid_mask is None:
                assert seg_label is not None
                valid_mask = seg_label != self.ignore_index

            flat_losses = losses.view(-1)
            sort_losses, sort_indices = flat_losses[valid_mask.view(-1)].sort()

            weights = torch.zeros(sort_losses.size())
            ext_module.compute_weights(
                sort_losses.size(0),
                sort_losses.cpu(),
                sort_indices.cpu(),
                weights.cpu(),
                self.ratio,
                self.p
            )

            seg_weight = torch.zeros_like(losses)
            seg_weight[valid_mask] = float(sort_losses.size(0)) * weights.to(losses.device)

            return seg_weight
