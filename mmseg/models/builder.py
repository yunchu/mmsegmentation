# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS
PARAMS_MANAGERS = MODELS
SCALAR_SCHEDULERS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg, ignore_index=255):
    """Build loss."""
    return LOSSES.build(cfg, default_args=dict(ignore_index=ignore_index))


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '

    return SEGMENTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_params_manager(cfg):
    return PARAMS_MANAGERS.build(cfg)


def build_scheduler(cfg, default_value=None):
    if cfg is None:
        if default_value is not None:
            assert isinstance(default_value, (int, float))
            cfg = dict(type='ConstantScalarScheduler', scale=float(default_value))
        else:
            return None
    elif isinstance(cfg, (int, float)):
        cfg = dict(type='ConstantScalarScheduler', scale=float(cfg))

    return SCALAR_SCHEDULERS.build(cfg)
