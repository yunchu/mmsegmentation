# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch

from mmcv.parallel import MMDataParallel
from mmcv.parallel import MMDistributedDataParallel
from mmseg.parallel import MMDataCPU


def prepare_mmseg_model_for_execution(model, cfg, distributed=False):
    """
    Prepare model for execution.
    Return model import ast, MMDataParallel or MMDataCPU.

    :param model: Model.
    :param cfg: training mmdet config.
    :param distributed: Enable distributed training mode.
    :return:
    """
    if torch.cuda.is_available():
        if distributed:
            # put model on gpus
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDataParallel(
                model.cuda(cfg.gpu_ids[0]),
                device_ids=[0]
            )
    else:
        model = MMDataCPU(model)
    return model
