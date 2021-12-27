# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class KvasirDataset(CustomDataset):
    """HRF dataset.

    In segmentation map annotation for Kvasir-Instrument, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The ``img_suffix`` is
    fixed to '.jpg' and ``seg_map_suffix`` is fixed to'.png'.
    """

    CLASSES = ('background', 'target')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, **kwargs):
        super(KvasirDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
