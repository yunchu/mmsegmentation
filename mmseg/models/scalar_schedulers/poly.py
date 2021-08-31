import numpy as np

from ..builder import SCALAR_SCHEDULERS
from .base import BaseScalarScheduler


@SCALAR_SCHEDULERS.register_module()
class PolyScalarScheduler(BaseScalarScheduler):
    def __init__(self, start_scale, end_scale, num_iters, power=1.2):
        super(PolyScalarScheduler, self).__init__()

        self._start_s = start_scale
        assert self._start_s >= 0.0
        self._end_s = end_scale
        assert self._end_s >= 0.0
        self._num_iters = num_iters
        assert self._num_iters > 0
        self._power = power
        assert self._power >= 0.0

    def _get_value(self, step):
        if step is None:
            return float(self._end_s)

        if step < self._num_iters:
            factor = (self._end_s - self._start_s) / (1.0 - self._power)
            var_a = factor / (self._num_iters ** self._power)
            var_b = -factor * self._power / float(self._num_iters)

            out_value = var_a * np.power(step, self._power) + var_b * step + self._start_s
        else:
            out_value = self._end_s

        return out_value
