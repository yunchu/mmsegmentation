from .config_utils import (patch_config,
                           set_hyperparams,
                           set_values_as_default)
from .configuration import OTESegmentationConfig
# from .openvino_task import OpenVINOSegmentationTask
from .ote_utils import get_task_class, load_template
from .task import OTESegmentationTask

__all__ = [
    'patch_config',
    'set_hyperparams',
    'set_values_as_default',
    'get_task_class',
    'load_template',
    'OTESegmentationConfig',
    'OTESegmentationTask',
    # 'OpenVINOSegmentationTask',
]
