# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.


import copy
import glob
import logging
import os
import tempfile
from collections import defaultdict
from typing import List, Optional

from mmcv import Config, ConfigDict
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label import LabelEntity
from ote_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback

from .configuration import OTESegmentationConfig


logger = logging.getLogger(__name__)


def is_epoch_based_runner(runner_config: ConfigDict):
    return 'Epoch' in runner_config.type


def patch_config(config: Config,
                 work_dir: str,
                 labels: List[LabelEntity],
                 random_seed: Optional[int] = None,
                 distributed=False):
    # Set runner if not defined.
    if 'runner' not in config:
        config.runner = {'type': 'IterBasedRunner'}

    # Check that there is no conflict in specification of number of training epochs.
    # Move global definition of epochs inside runner config.
    if 'total_epochs' in config:
        if is_epoch_based_runner(config.runner):
            if config.runner.max_epochs != config.total_epochs:
                logger.warning('Conflicting declaration of training epochs number.')
            config.runner.max_epochs = config.total_epochs
        else:
            logger.warning(f'Total number of epochs set for an iteration based runner {config.runner.type}.')

        remove_from_config(config, 'total_epochs')

    # Change runner's type.
    if is_epoch_based_runner(config.runner):
        logger.info(f'Replacing runner from {config.runner.type} to EpochRunnerWithCancel.')
        config.runner.type = 'EpochRunnerWithCancel'
    else:
        logger.info(f'Replacing runner from {config.runner.type} to IterBasedRunnerWithCancel.')
        config.runner.type = 'IterBasedRunnerWithCancel'

    # Add training cancelation hook.
    if 'custom_hooks' not in config:
        config.custom_hooks = []
    if 'CancelTrainingHook' not in {hook.type for hook in config.custom_hooks}:
        config.custom_hooks.append({'type': 'CancelTrainingHook'})

    # Remove high level data pipelines definition leaving them only inside `data` section.
    remove_from_config(config, 'train_pipeline')
    remove_from_config(config, 'test_pipeline')

    # Patch data pipeline, making it OTE-compatible.
    patch_datasets(config)

    if 'log_config' not in config:
        config.log_config = ConfigDict()
    config.log_config.hooks = []

    if 'evaluation' not in config:
        config.evaluation = ConfigDict()
    evaluation_metric = config.evaluation.get('metric')
    if evaluation_metric is not None:
        config.evaluation.save_best = evaluation_metric

    if 'checkpoint_config' not in config:
        config.checkpoint_config = ConfigDict()
    config.checkpoint_config.max_keep_ckpts = 5
    config.checkpoint_config.interval = config.evaluation.get('interval', 1)

    label_names = ['background'] + [label.name for label in labels]
    set_data_classes(config, label_names)
    set_num_classes(config, len(label_names))

    set_distributed_mode(config, distributed)
    remove_from_config(config, 'norm_cfg')

    config.gpu_ids = range(1)
    config.work_dir = work_dir
    config.seed = random_seed


def set_hyperparams(config: Config, hyperparams: OTESegmentationConfig):
    config.data.samples_per_gpu = int(hyperparams.learning_parameters.batch_size)
    config.data.workers_per_gpu = int(hyperparams.learning_parameters.num_workers)
    config.optimizer.lr = float(hyperparams.learning_parameters.learning_rate)

    # set proper number of iterations
    config.params_config.iters = int(hyperparams.learning_parameters.learning_rate_fixed_iters)
    config.lr_config.fixed_iters = int(hyperparams.learning_parameters.learning_rate_fixed_iters)
    config.lr_config.warmup_iters = int(hyperparams.learning_parameters.learning_rate_warmup_iters)
    total_iterations = int(hyperparams.learning_parameters.num_iters)
    if is_epoch_based_runner(config.runner):
        init_num_iterations = config.runner.max_epochs
        config.runner.max_epochs = total_iterations
    else:
        init_num_iterations = config.runner.max_iters
        config.runner.max_iters = total_iterations

    # estimate the schedule scale
    schedule_scale = float(total_iterations) / float(init_num_iterations)

    # rescale number of iterations for lr scheduler
    if config.lr_config.policy == 'customstep':
        config.lr_config.step = [int(schedule_scale * step) for step in config.lr_config.step]
    elif config.lr_config.policy == 'customcos':
        config.lr_config.periods = [int(schedule_scale * period) for period in config.lr_config.periods]

    # rescale number of iterations for
    for head_type in ['decode_head', 'auxiliary_head']:
        if not hasattr(config.model, head_type):
            continue

        heads = config.model[head_type]
        if not isinstance(heads, (tuple, list)):
            heads = [heads]

        for head in heads:
            losses = head.loss_decode
            if not isinstance(losses, (tuple, list)):
                losses = [losses]

            for loss in losses:
                for target_attr in ['scale_cfg', 'conf_penalty_weight']:
                    if not hasattr(loss, target_attr):
                        continue

                    if not hasattr(loss[target_attr], 'num_iters'):
                        continue

                    loss[target_attr].num_iters = int(schedule_scale * loss[target_attr].num_iters)

            head.loss_decode = losses

        config.model[head_type] = heads


def prepare_for_testing(config: Config, dataset: DatasetEntity) -> Config:
    config = copy.deepcopy(config)
    config.data.test.ote_dataset = dataset
    return config


def prepare_for_training(config: Config, train_dataset: DatasetEntity, val_dataset: DatasetEntity,
                         time_monitor: TimeMonitorCallback, learning_curves: defaultdict) -> Config:
    config = copy.deepcopy(config)
    prepare_work_dir(config)
    config.data.val.ote_dataset = val_dataset
    if 'ote_dataset' in config.data.train:
        config.data.train.ote_dataset = train_dataset
    else:
        config.data.train.dataset.ote_dataset = train_dataset
    config.custom_hooks.append({'type': 'OTEProgressHook', 'time_monitor': time_monitor, 'verbose': True})
    config.log_config.hooks.append({'type': 'OTELoggerHook', 'curves': learning_curves})
    return config


def config_to_string(config: Config) -> str:
    """
    Convert a full mmsegmentation config to a string.

    :param config: configuration object to convert
    :return str: string representation of the configuration
    """

    config_copy = copy.deepcopy(config)
    # Clean config up by removing dataset as this causes the pretty text parsing to fail.
    config_copy.data.test.ote_dataset = None
    config_copy.data.val.ote_dataset = None
    if 'ote_dataset' in config_copy.data.train:
        config_copy.data.train.ote_dataset = None
    else:
        config_copy.data.train.dataset.ote_dataset = None
    return Config(config_copy).pretty_text


def config_from_string(config_string: str) -> Config:
    """
    Generate an mmsegmentation config dict object from a string.

    :param config_string: string to parse
    :return config: configuration object
    """

    with tempfile.NamedTemporaryFile('w', suffix='.py') as temp_file:
        temp_file.write(config_string)
        temp_file.flush()
        return Config.fromfile(temp_file.name)


def save_config_to_file(config: Config):
    """ Dump the full config to a file. Filename is 'config.py', it is saved in the current work_dir. """

    filepath = os.path.join(config.work_dir, 'config.py')
    config_string = config_to_string(config)
    with open(filepath, 'w') as f:
        f.write(config_string)


def prepare_work_dir(config: Config) -> str:
    base_work_dir = config.work_dir
    checkpoint_dirs = glob.glob(os.path.join(base_work_dir, "checkpoints_round_*"))
    train_round_checkpoint_dir = os.path.join(base_work_dir, f"checkpoints_round_{len(checkpoint_dirs)}")
    os.makedirs(train_round_checkpoint_dir)
    logger.info(f"Checkpoints and logs for this training run are stored in {train_round_checkpoint_dir}")
    config.work_dir = train_round_checkpoint_dir
    if 'meta' not in config.runner:
        config.runner.meta = ConfigDict()
    config.runner.meta.exp_name = f"train_round_{len(checkpoint_dirs)}"
    # Save training config for debugging. It is saved in the checkpoint dir for this training round.
    save_config_to_file(config)
    return train_round_checkpoint_dir


def set_distributed_mode(config: Config, distributed: bool):
    if distributed:
        return

    norm_cfg = {'type': 'BN', 'requires_grad': True}

    def _replace_syncbn(_node, _norm_cfg):
        if _node.norm_cfg.type != 'SyncBN':
            return

        _node.norm_cfg = _norm_cfg

    config.model.backbone.norm_cfg = norm_cfg
    for head_type in ('decode_head', 'auxiliary_head'):
        head = config.model.get(head_type, None)
        if head is None:
            continue

        if isinstance(head, (tuple, list)):
            for sub_head in head:
                _replace_syncbn(sub_head, norm_cfg)
        else:
            _replace_syncbn(head, norm_cfg)


def set_data_classes(config: Config, label_names: List[str]):
    # Save labels in data configs.
    for subset in ('train', 'val', 'test'):
        cfg = config.data[subset]
        if cfg.type == 'RepeatDataset':
            cfg.dataset.classes = label_names
        else:
            cfg.classes = label_names
        config.data[subset].classes = label_names


def set_num_classes(config: Config, num_classes: int):
    assert num_classes > 1

    for head_type in ('decode_head', 'auxiliary_head'):
        head = config.model.get(head_type, None)
        if head is None:
            continue

        if isinstance(head, (tuple, list)):
            for sub_head in head:
                sub_head.num_classes = num_classes
        else:
            head.num_classes = num_classes


def patch_datasets(config: Config):
    assert 'data' in config
    for subset in ('train', 'val', 'test'):
        cfg = config.data[subset]
        if cfg.type == 'RepeatDataset':
            cfg = cfg.dataset

        cfg.type = 'OTEDataset'
        cfg.ote_dataset = None

        remove_from_config(cfg, 'ann_dir')
        remove_from_config(cfg, 'img_dir')

        for pipeline_step in cfg.pipeline:
            if pipeline_step.type == 'LoadImageFromFile':
                pipeline_step.type = 'LoadImageFromOTEDataset'
            elif pipeline_step.type == 'LoadAnnotations':
                pipeline_step.type = 'LoadAnnotationFromOTEDataset'


def remove_from_config(config, key: str):
    if key in config:
        if isinstance(config, Config):
            del config._cfg_dict[key]
        elif isinstance(config, ConfigDict):
            del config[key]
        else:
            raise ValueError(f'Unknown config type {type(config)}')


def set_values_as_default(parameters):
    for v in parameters.values():
        if isinstance(v, dict) and 'value' not in v:
            set_values_as_default(v)
        elif isinstance(v, dict) and 'value' in v:
            if v['value'] != v['default_value']:
                v['value'] = v['default_value']
