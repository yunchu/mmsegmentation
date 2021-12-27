# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .base_pixel_sampler import BasePixelSampler
from .ohem_pixel_sampler import OHEMPixelSampler
from .class_weighting_pixel_sampler import ClassWeightingPixelSampler
from .max_pooling_pixel_sampler import MaxPoolingPixelSampler

__all__ = [
    'BasePixelSampler',
    'OHEMPixelSampler',
    'ClassWeightingPixelSampler',
    'MaxPoolingPixelSampler',
]
