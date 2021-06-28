from abc import ABCMeta, abstractmethod


class BasePixelSampler(metaclass=ABCMeta):
    """Base class of pixel sampler."""

    def __init__(self, context, **kwargs):
        self.context = context
        assert self.context is not None

    @abstractmethod
    def sample(self, seg_logit, seg_label):
        """Placeholder for sample function."""
