import numpy as np

from ..builder import SCALAR_SCHEDULERS
from .base import BaseScalarScheduler


@SCALAR_SCHEDULERS.register_module()
class StepScalarScheduler(BaseScalarScheduler):
    def __init__(self, scales, num_iters):
        super(StepScalarScheduler, self).__init__()

        assert len(scales) == len(num_iters) + 1
        assert len(scales) > 0

        self._scales = list(scales)
        self._iter_ranges = list(num_iters) + [np.iinfo(np.int32).max]

    def _get_scale(self, step):
        if step is None:
            return float(self._scales[-1])

        out_scale_idx = 0
        for iter_range in self._iter_ranges:
            if step < iter_range:
                break

            out_scale_idx += 1

        return float(self._scales[out_scale_idx])
