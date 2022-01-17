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
import io
import logging
import os
import shutil
import tempfile
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import Config
from ote_sdk.utils.segmentation_utils import (create_hard_prediction_from_soft_prediction,
                                              create_annotation_from_segmentation_map)
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.inference_parameters import default_progress_callback as default_infer_progress_callback
from ote_sdk.entities.model import ModelEntity, ModelFormat, ModelOptimizationType, ModelPrecision
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.tensor import TensorEntity
from ote_sdk.serialization.label_mapper import label_schema_to_bytes
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload


from mmseg.apis import export_model
from mmseg.apis.ote.apis.segmentation.config_utils import (patch_config,
                                                           prepare_for_testing,
                                                           set_hyperparams)
from mmseg.apis.ote.apis.segmentation.configuration import OTESegmentationConfig
from mmseg.apis.ote.apis.segmentation.ote_utils import InferenceProgressCallback
from mmseg.core.evaluation.metrics import f_score
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.parallel import MMDataCPU

logger = logging.getLogger(__name__)


class OTESegmentationInferenceTask(IInferenceTask, IExportTask, IEvaluationTask, IUnload):
    task_environment: TaskEnvironment

    def __init__(self, task_environment: TaskEnvironment):
        """"
        Task for training semantic segmentation models using OTESegmentation.

        """

        logger.info(f"Loading OTESegmentationTask.")
        self._scratch_space = tempfile.mkdtemp(prefix="ote-seg-scratch-")
        logger.info(f"Scratch space created at {self._scratch_space}")

        self._task_environment = task_environment

        self._model_name = task_environment.model_template.name
        self._labels = task_environment.get_labels(include_empty=False)

        template_file_path = task_environment.model_template.model_template_path

        # Get and prepare mmseg config.
        self._base_dir = os.path.abspath(os.path.dirname(template_file_path))
        config_file_path = os.path.join(self._base_dir, "model.py")
        self._config = Config.fromfile(config_file_path)

        distributed = torch.distributed.is_initialized()
        patch_config(self._config, self._scratch_space, self._labels,
                     random_seed=42, distributed=distributed)
        set_hyperparams(self._config, self._hyperparams)

        # Set default model attributes.
        self._optimization_methods = []
        self._precision = [ModelPrecision.FP32]
        self._optimization_type = ModelOptimizationType.MO

        # Create and initialize PyTorch model.
        self._model = self._load_model(task_environment.model)

        # Extra control variables.
        self._training_work_dir = None
        self._is_training = False
        self._should_stop = False

    @property
    def _hyperparams(self):
        return self._task_environment.get_hyper_parameters(OTESegmentationConfig)

    def _load_model(self, model: ModelEntity):
        if model is not None:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device('cpu'))

            model = self._create_model(self._config, from_scratch=True)

            try:
                model.load_state_dict(model_data['model'])
                logger.info(f"Loaded model weights from Task Environment")
                logger.info(f"Model architecture: {self._model_name}")
            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                    from ex
        else:
            # If there is no trained model yet, create model with pretrained weights as defined in the model config
            # file.
            model = self._create_model(self._config, from_scratch=False)
            logger.info(f"No trained model in project yet. Created new model with '{self._model_name}' "
                        f"architecture and general-purpose pretrained weights.")
        return model

    @staticmethod
    def _create_model(config: Config, from_scratch: bool = False):
        """
        Creates a model, based on the configuration in config

        :param config: mmsegmentation configuration from which the model has to be built
        :param from_scratch: bool, if True does not load any weights

        :return model: ModelEntity in training mode
        """

        model_cfg = copy.deepcopy(config.model)

        init_from = None if from_scratch else config.get('load_from', None)
        logger.warning(f"Init from: {init_from}")

        if init_from is not None:
            # No need to initialize backbone separately, if all weights are provided.
            model_cfg.pretrained = None
            logger.warning('build segmentor')
            model = build_segmentor(model_cfg)

            # Load all weights.
            logger.warning('load checkpoint')
            load_checkpoint(model, init_from, map_location='cpu', strict=False)
        else:
            logger.warning('build segmentor')
            model = build_segmentor(model_cfg)

        return model

    def infer(self, dataset: DatasetEntity,
              inference_parameters: Optional[InferenceParameters] = None) -> DatasetEntity:
        """ Analyzes a dataset using the latest inference model. """

        set_hyperparams(self._config, self._hyperparams)

        # There is no need to have many workers for a couple of images.
        self._config.data.workers_per_gpu = max(min(self._config.data.workers_per_gpu, len(dataset) - 1), 0)

        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
            is_evaluation = inference_parameters.is_evaluation
        else:
            update_progress_callback = default_infer_progress_callback
            is_evaluation = False
        dump_features = not is_evaluation

        time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)

        def pre_hook(module, input):
            time_monitor.on_test_batch_begin(None, None)

        def hook(module, input, output):
            time_monitor.on_test_batch_end(None, None)

        pre_hook_handle = self._model.register_forward_pre_hook(pre_hook)
        hook_handle = self._model.register_forward_hook(hook)

        self._infer_segmentor(self._model, self._config, dataset, eval=False,
                              output_logits=True, dump_features=True, dump_predictions_to_dataset=True)

        pre_hook_handle.remove()
        hook_handle.remove()

        return dataset

    def _infer_segmentor(self,
                         model: torch.nn.Module, config: Config, dataset: DatasetEntity,
                         eval: Optional[bool] = False, metric_name: Optional[str] = 'mDice',
                         output_logits: bool = False, dump_features: bool = True,
                         dump_predictions_to_dataset: bool = False) -> Optional[float]:
        model.eval()

        test_config = prepare_for_testing(config, dataset)
        mm_val_dataset = build_dataset(test_config.data.test)

        batch_size = 1
        mm_val_dataloader = build_dataloader(mm_val_dataset,
                                             samples_per_gpu=batch_size,
                                             workers_per_gpu=test_config.data.workers_per_gpu,
                                             num_gpus=1,
                                             dist=False,
                                             shuffle=False)
        if torch.cuda.is_available():
            eval_model = MMDataParallel(model.cuda(test_config.gpu_ids[0]),
                                        device_ids=test_config.gpu_ids)
        else:
            eval_model = MMDataCPU(model)

        feature_vector = None

        def dump_features_hook(mod, inp, out):
            with torch.no_grad():
                feature_map = out[0]
                feature_map = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
                assert feature_map.size(0) == 1
            nonlocal feature_vector
            feature_vector = feature_map.view(-1).detach().cpu().numpy()

        def dummy_dump_features_hook(mod, inp, out):
            pass

        hook = dump_features_hook if dump_features else dummy_dump_features_hook

        stats = defaultdict(float)
        
        label_dictionary = {
            i + 1: self._labels[i] for i in range(len(self._labels))
        }
            
        # Use a single gpu for testing. Set in both mm_val_dataloader and eval_model
        with eval_model.module.backbone.register_forward_hook(hook):
            for i, (data, dataset_item) in enumerate(zip(mm_val_dataloader, dataset)):
                with torch.no_grad():
                    result = eval_model(return_loss=False, output_logits=output_logits, **data)
                assert len(result) == 1
                if eval:
                    for k, v in mm_val_dataset.compute_stat_per_frame(result[0], frame_id=i).items():
                        stats[k] += v

                if dump_predictions_to_dataset:
                    soft_prediction = np.transpose(result[0], axes=(1, 2, 0))
                    hard_prediction = create_hard_prediction_from_soft_prediction(
                        soft_prediction=soft_prediction,
                        soft_threshold=self._hyperparams.postprocessing.soft_threshold,
                        blur_strength=self._hyperparams.postprocessing.blur_strength,
                    )
                    annotations = create_annotation_from_segmentation_map(
                        hard_prediction=hard_prediction,
                        soft_prediction=soft_prediction,
                        label_map=label_dictionary,
                    )
                    dataset_item.append_annotations(annotations=annotations)
                    if feature_vector is not None:
                        active_score = TensorEntity(name="representation_vector", numpy=feature_vector)
                        dataset_item.append_metadata_item(active_score, model=self._task_environment.model)
                    if dump_features:
                        for label_index, label in label_dictionary.items():
                            if label_index == 0:
                                continue
                            if len(soft_prediction.shape) == 3:
                                current_label_soft_prediction = soft_prediction[:, :, label_index]
                            else:
                                current_label_soft_prediction = soft_prediction
                            min_soft_score = np.min(current_label_soft_prediction)
                            max_soft_score = np.max(current_label_soft_prediction)
                            factor = 255.0 / (max_soft_score - min_soft_score + 1e-12)
                            result_media_numpy = (factor * (current_label_soft_prediction - min_soft_score)).astype(np.uint8)
                            result_media = ResultMediaEntity(name=f'{label.name}',
                                                            type='Soft Prediction',
                                                            label=label,
                                                            annotation_scene=dataset_item.annotation_scene,
                                                            roi=dataset_item.roi,
                                                            numpy=result_media_numpy)
                            dataset_item.append_metadata_item(result_media, model=self._task_environment.model)

        metric = None
        if eval:
            assert not output_logits
            if metric_name == 'mIoU':
                metric = stats['total_area_intersect'] / stats['total_area_union']
            elif metric_name == 'mDice':
                metric = 2 * stats['total_area_intersect'] / (stats['total_area_pred_label'] + stats['total_area_label'])
            elif metric_name == 'mFscore':
                beta = 1
                precision = stats['total_area_intersect'] / stats['total_area_pred_label']
                recall = stats['total_area_intersect'] / stats['total_area_label']
                metric = np.array([f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            else:
                raise ValueError(f"Ivalid metric_name: {metric_name}")

            metric = np.round(np.nanmean(metric) * 100.0, 2) / 100.0

        return metric

    def evaluate(self, output_result_set: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """ Computes performance on a resultset """

        logger.info('Computing mDice')
        metrics = MetricsHelper.compute_dice_averaged_over_pixels(
            output_result_set
        )
        logger.info(f"mDice after evaluation: {metrics.overall_dice.value}")

        output_result_set.performance = metrics.get_performance()

    @staticmethod
    def _is_docker():
        """
        Checks whether the task runs in docker container

        :return bool: True if task runs in docker
        """
        path = '/proc/self/cgroup'
        is_in_docker = False
        if os.path.isfile(path):
            with open(path) as f:
                is_in_docker = is_in_docker or any('docker' in line for line in f)
        is_in_docker = is_in_docker or os.path.exists('/.dockerenv')
        return is_in_docker

    def unload(self):
        """
        Unload the task
        """

        self._delete_scratch_space()

        if self._is_docker():
            logger.warning("Got unload request. Unloading models. Throwing Segmentation Fault on purpose")
            import ctypes
            ctypes.string_at(0)
        else:
            logger.warning("Got unload request, but not on Docker. Only clearing CUDA cache")
            torch.cuda.empty_cache()
            logger.warning(f"Done unloading. "
                           f"Torch is still occupying {torch.cuda.memory_allocated()} bytes of GPU memory")

    def export(self, export_type: ExportType, output_model: ModelEntity):
        assert export_type == ExportType.OPENVINO

        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = self._optimization_type

        with tempfile.TemporaryDirectory() as tempdir:
            optimized_model_dir = os.path.join(tempdir, "export")
            logger.info(f'Optimized model will be temporarily saved to "{optimized_model_dir}"')
            os.makedirs(optimized_model_dir, exist_ok=True)

            try:
                from torch.jit._trace import TracerWarning
                warnings.filterwarnings("ignore", category=TracerWarning)

                if torch.cuda.is_available():
                    model = self._model.cuda(self._config.gpu_ids[0])
                else:
                    model = self._model.cpu()

                export_model(model,
                             self._config,
                             tempdir,
                             target='openvino',
                             output_logits=True,
                             input_format='bgr')  # ote expects RGB but mmseg uses BGR, so invert it

                bin_file = [f for f in os.listdir(tempdir) if f.endswith('.bin')][0]
                xml_file = [f for f in os.listdir(tempdir) if f.endswith('.xml')][0]
                with open(os.path.join(tempdir, bin_file), "rb") as f:
                    output_model.set_data("openvino.bin", f.read())
                with open(os.path.join(tempdir, xml_file), "rb") as f:
                    output_model.set_data("openvino.xml", f.read())
                output_model.precision = self._precision
                output_model.optimization_methods = self._optimization_methods
            except Exception as ex:
                raise RuntimeError("Optimization was unsuccessful.") from ex

        output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))

    def _delete_scratch_space(self):
        """
        Remove model checkpoints and mmseg logs
        """

        if os.path.exists(self._scratch_space):
            shutil.rmtree(self._scratch_space, ignore_errors=False)
