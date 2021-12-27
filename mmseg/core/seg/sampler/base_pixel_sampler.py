# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from abc import ABCMeta, abstractmethod


class BasePixelSampler(metaclass=ABCMeta):
    """Base class of pixel sampler."""

    def __init__(self, ignore_index=255, **kwargs):
        self.ignore_index = ignore_index

    @abstractmethod
    def _sample(self, losses=None, seg_logit=None, seg_label=None, valid_mask=None):
        """Placeholder for sample function."""

    def __call__(self, *args, **kwargs):
        return self._sample(*args, **kwargs).float()
