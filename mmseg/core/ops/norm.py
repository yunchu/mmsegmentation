# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalContrastNormalization(nn.Module):
    def __init__(self, k=3, sigma=2.0, num_channels=3, use_divisor=True):
        super().__init__()

        assert k % 2 == 1
        assert num_channels > 0
        assert eps >= 0.0

        gaussian_filter = self._gaussian_filter([k, k], num_channels, sigma)
        self.register_buffer('weight', torch.from_numpy(gaussian_filter))

        self.padding = (k - 1) // 2
        self.use_divisor = use_divisor

    @staticmethod
    def _gaussian_filter(kernel, channels, sigma):
        def _gauss_value(x, y):
            Z = 2 * np.pi * sigma ** 2
            return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

        anchor = (np.floor(0.5 * kernel[0]),
                  np.floor(0.5 * kernel[1]))

        gaussian_filter = np.empty(kernel, dtype=np.float32)
        for i in range(kernel[0]):
            for j in range(kernel[1]):
                gaussian_filter[i, j] = _gauss_value(i - anchor[0], j - anchor[1])

        gaussian_filter = np.tile(gaussian_filter.reshape([1, 1] + kernel), [1, channels, 1, 1])
        gaussian_filter = gaussian_filter / np.sum(gaussian_filter)

        return gaussian_filter

    def forward(self, x):
        local_mean = F.conv2d(x, self.weight, padding=self.padding)
        y = x - local_mean

        if self.use_divisor:
            sqr_y = torch.square(y)
            local_sqr_mean = F.conv2d(sqr_y, self.weight, padding=self.padding)
            sigma = torch.sqrt(local_sqr_mean)

            mean_sigma = torch.mean(sigma, dim=(2, 3), keepdim=True)
            fixed_sigma = torch.maximum(sigma, mean_sigma)

            out = y / fixed_sigma
        else:
            out = y

        return out
