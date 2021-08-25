from abc import ABCMeta, abstractmethod

import torch.nn as nn

from mmseg.core import build_pixel_sampler


class BaseWeightedLoss(nn.Module, metaclass=ABCMeta):
    """Base class for loss.

    All subclass should overwrite the ``_forward()`` method which returns the
    normal loss without loss weights.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, ignore_index=255, sampler=None, **kwargs):
        super().__init__()

        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

        self.sampler = None
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, ignore_index=ignore_index)

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call.

        Args:
            *args: The positional arguments for the corresponding
                loss.
            **kwargs: The keyword arguments for the corresponding
                loss.

        Returns:
            torch.Tensor: The calculated loss.
        """

        loss = self._forward(*args, **kwargs)
        out = self.loss_weight * loss

        return out
