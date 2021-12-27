# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

COMPOSE_TYPES = ['ProbCompose']
TARGET_TYPES = ['CrossNorm', 'MixUp']


def _propagate_data_pipeline(pipeline, root_dir):
    for stage in pipeline:
        if 'type' not in stage:
            continue

        if stage['type'] in COMPOSE_TYPES:
            _propagate_data_pipeline(stage['transforms'], root_dir)
        elif stage['type'] in TARGET_TYPES:
            stage['root_dir'] = root_dir


def _add2dataset(cfg, root_dir):
    if 'dataset' in cfg:
        cfg.dataset.data_root = root_dir
        _propagate_data_pipeline(cfg.dataset.pipeline, root_dir)
    else:
        cfg.data_root = root_dir
        _propagate_data_pipeline(cfg.pipeline, root_dir)


def propagate_root_dir(cfg, root_dir=None):
    if root_dir is not None:
        cfg.data_root = root_dir

    assert cfg.data_root is not None and cfg.data_root != ''

    for trg_data_type in ['train', 'val', 'test']:
        _add2dataset(cfg.data[trg_data_type], cfg.data_root)

    return cfg
