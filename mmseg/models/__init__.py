# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, HEADS, LOSSES, SEGMENTORS,
                      build_backbone, build_head, build_loss,
                      build_segmentor, build_params_manager,
                      build_neck, build_scheduler)
from .decode_heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403
from .params import *  # noqa: F401,F403
from .scalar_schedulers import *  # noqa: F401,F403

__all__ = [
    'BACKBONES', 'HEADS', 'LOSSES', 'SEGMENTORS',
    'build_backbone', 'build_head', 'build_loss', 'build_neck', 'build_segmentor',
    'build_params_manager', 'build_scheduler',
]
