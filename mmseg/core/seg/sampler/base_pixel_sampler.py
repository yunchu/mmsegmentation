from abc import ABCMeta, abstractmethod


class BasePixelSampler(metaclass=ABCMeta):
    """Base class of pixel sampler."""

    def __init__(self, ignore_index=255, **kwargs):
        self.ignore_index = ignore_index

    @abstractmethod
    def _sample(self, losses, valid_mask):
        """Placeholder for sample function."""

    def __call__(self, *args, **kwargs):
        return self._sample(*args, **kwargs).float()
