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

import attr
import logging
import inspect
import json
import os
from shutil import copyfile, copytree
import sys
import subprocess  # nosec
import tempfile
from addict import Dict as ADDict
from typing import Any, Dict, Tuple, Optional, Union
from zipfile import ZipFile

import numpy as np

from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.annotation import AnnotationSceneEntity
from ote_sdk.entities.inference_parameters import InferenceParameters, default_progress_callback
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod
)
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.exportable_code.inference import BaseInferencer
from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import SegmentationToAnnotationConverter
from ote_sdk.usecases.exportable_code.utils import set_proper_git_commit_hash
import ote_sdk.usecases.exportable_code.demo as demo
from ote_sdk.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.optimization_interface import IOptimizationTask, OptimizationType

from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline
from ote_sdk.serialization.label_mapper import LabelSchemaMapper, label_schema_to_bytes

from .configuration import OTESegmentationConfig
from openvino.model_zoo.model_api.models import Model
from openvino.model_zoo.model_api.adapters import create_core, OpenvinoAdapter
from . import model_wrappers
logger = logging.getLogger(__name__)


class OpenVINOSegmentationInferencer(BaseInferencer):
    def __init__(
        self,
        hparams: OTESegmentationConfig,
        label_schema: LabelSchemaEntity,
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):
        """
        Inferencer implementation for OTESegmentation using OpenVINO backend.

        :param hparams: Hyper parameters that the model should use.
        :param label_schema: LabelSchemaEntity that was used during model training.
        :param model_file: Path to model to load, `.xml`, `.bin` or `.onnx` file.
        :param device: Device to run inference on, such as CPU, GPU or MYRIAD. Defaults to "CPU".
        :param num_requests: Maximum number of requests that the inferencer can make.
            Good value is the number of available cores. Defaults to 1.
        """

        model_adapter = OpenvinoAdapter(create_core(), model_file, weight_file, device=device, max_num_requests=num_requests)
        self.configuration = {**attr.asdict(hparams.postprocessing,
                              filter=lambda attr, value: attr.name not in ['header', 'description', 'type', 'visible_in_ui', 'class_name'])}
        self.model = Model.create_model(hparams.postprocessing.class_name.value, model_adapter, self.configuration, preload=True)
        self.converter = SegmentationToAnnotationConverter(label_schema)

    def pre_process(self, image: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        return self.model.preprocess(image)

    def post_process(self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> AnnotationSceneEntity:
        hard_prediction = self.model.postprocess(prediction, metadata)

        return self.converter.convert_to_annotation(hard_prediction, metadata)

    def forward(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.model.infer_sync(inputs)


class OTEOpenVinoDataLoader(DataLoader):
    def __init__(self, dataset: DatasetEntity, inferencer: BaseInferencer):
        self.dataset = dataset
        self.inferencer = inferencer

    def __getitem__(self, index):
        image = self.dataset[index].numpy
        annotation = self.dataset[index].annotation_scene
        inputs, metadata = self.inferencer.pre_process(image)

        return (index, annotation), inputs, metadata

    def __len__(self):
        return len(self.dataset)


class OpenVINOSegmentationTask(IDeploymentTask, IInferenceTask, IEvaluationTask, IOptimizationTask):
    def __init__(self,
                 task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.model = self.task_environment.model
        self.model_name = task_environment.model_template.model_template_id
        self.inferencer = self.load_inferencer()
        template_file_path = task_environment.model_template.model_template_path
        self._base_dir = os.path.abspath(os.path.dirname(template_file_path))

    @property
    def hparams(self):
        return self.task_environment.get_hyper_parameters(OTESegmentationConfig)

    def load_inferencer(self) -> OpenVINOSegmentationInferencer:
        return OpenVINOSegmentationInferencer(self.hparams,
                                              self.task_environment.label_schema,
                                              self.model.get_data("openvino.xml"),
                                              self.model.get_data("openvino.bin"))

    def infer(self,
              dataset: DatasetEntity,
              inference_parameters: Optional[InferenceParameters] = None) -> DatasetEntity:
        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
        dataset_size = len(dataset)
        for i, dataset_item in enumerate(dataset, 1):
            predicted_scene = self.inferencer.predict(dataset_item.numpy)
            dataset_item.append_annotations(predicted_scene.annotations)
            update_progress_callback(int(i / dataset_size * 100))
        return dataset

    def evaluate(self,
                 output_result_set: ResultSetEntity,
                 evaluation_metric: Optional[str] = None):
        logger.info('Computing mDice')
        metrics = MetricsHelper.compute_dice_averaged_over_pixels(
            output_result_set
        )
        logger.info(f"mDice after evaluation: {metrics.overall_dice.value}")

        output_result_set.performance = metrics.get_performance()

    def deploy(self,
               output_model: ModelEntity) -> None:
        logger.info('Deploying the model')

        work_dir = os.path.dirname(demo.__file__)
        model_file = inspect.getfile(type(self.inferencer.model))
        parameters = {}
        parameters['type_of_model'] = self.hparams.postprocessing.class_name.value
        parameters['converter_type'] = 'SEGMENTATION'
        parameters['model_parameters'] = self.inferencer.configuration
        parameters['model_parameters']['labels'] = LabelSchemaMapper.forward(self.task_environment.label_schema)
        name_of_package = "demo_package"
        with tempfile.TemporaryDirectory() as tempdir:
            copyfile(os.path.join(work_dir, "setup.py"), os.path.join(tempdir, "setup.py"))
            copyfile(os.path.join(work_dir, "requirements.txt"), os.path.join(tempdir, "requirements.txt"))
            set_proper_git_commit_hash(os.path.join(tempdir, "requirements.txt"))
            copytree(os.path.join(work_dir, name_of_package), os.path.join(tempdir, name_of_package))
            config_path = os.path.join(tempdir, name_of_package, "config.json")
            print(parameters)
            with open(config_path, "w", encoding='utf-8') as f:
                json.dump(parameters, f, ensure_ascii=False, indent=4)
            # generate model.py
            if (inspect.getmodule(self.inferencer.model) in
               [module[1] for module in inspect.getmembers(model_wrappers, inspect.ismodule)]):
                copyfile(model_file, os.path.join(tempdir, name_of_package, "model.py"))
            # create wheel package
            subprocess.run([sys.executable, os.path.join(tempdir, "setup.py"), 'bdist_wheel',
                            '--dist-dir', tempdir, 'clean', '--all'])
            wheel_file_name = [f for f in os.listdir(tempdir) if f.endswith('.whl')][0]

            with ZipFile(os.path.join(tempdir, "openvino.zip"), 'w') as zip:
                zip.writestr(os.path.join("model", "model.xml"), self.model.get_data("openvino.xml"))
                zip.writestr(os.path.join("model", "model.bin"), self.model.get_data("openvino.bin"))
                zip.write(os.path.join(tempdir, "requirements.txt"), os.path.join("python", "requirements.txt"))
                zip.write(os.path.join(work_dir, "README.md"), os.path.join("python", "README.md"))
                zip.write(os.path.join(work_dir, "LICENSE"), os.path.join("python", "LICENSE"))
                zip.write(os.path.join(work_dir, "demo.py"), os.path.join("python", "demo.py"))
                zip.write(os.path.join(tempdir, wheel_file_name), os.path.join("python", wheel_file_name))
            with open(os.path.join(tempdir, "openvino.zip"), "rb") as file:
                output_model.exportable_code = file.read()
        logger.info('Deploying completed')

    def optimize(self,
                 optimization_type: OptimizationType,
                 dataset: DatasetEntity,
                 output_model: ModelEntity,
                 optimization_parameters: Optional[OptimizationParameters]):

        if optimization_type is not OptimizationType.POT:
            raise ValueError("POT is the only supported optimization type for OpenVino models")

        data_loader = OTEOpenVinoDataLoader(dataset, self.inferencer)

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            bin_path = os.path.join(tempdir, "model.bin")
            with open(xml_path, "wb") as f:
                f.write(self.model.get_data("openvino.xml"))
            with open(bin_path, "wb") as f:
                f.write(self.model.get_data("openvino.bin"))

            model_config = ADDict({
                'model_name': 'openvino_model',
                'model': xml_path,
                'weights': bin_path
            })

            model = load_model(model_config)

            if get_nodes_by_type(model, ['FakeQuantize']):
                raise RuntimeError("Model is already optimized by POT")

        engine_config = ADDict({
            'device': 'CPU'
        })

        optimization_config_path = os.path.join(self._base_dir, 'pot_optimization_config.json')
        if os.path.exists(optimization_config_path):
            with open(optimization_config_path) as f_src:
                algorithms = ADDict(json.load(f_src))['algorithms']
        else:
            algorithms = [
                ADDict({
                    'name': 'DefaultQuantization',
                    'params': {
                        'target_device': 'ANY'
                    }
                })
            ]
        for algo in algorithms:
            algo.params.stat_subset_size = self.hparams.pot_parameters.stat_subset_size
            algo.params.shuffle_data = True
            if 'Quantization' in algo['name']:
                algo.params.preset = self.hparams.pot_parameters.preset.name.lower()

        engine = IEEngine(config=engine_config, data_loader=data_loader, metric=None)

        pipeline = create_pipeline(algorithms, engine)

        compressed_model = pipeline.run(model)

        compress_model_weights(compressed_model)

        with tempfile.TemporaryDirectory() as tempdir:
            save_model(compressed_model, tempdir, model_name="model")
            with open(os.path.join(tempdir, "model.xml"), "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            with open(os.path.join(tempdir, "model.bin"), "rb") as f:
                output_model.set_data("openvino.bin", f.read())

        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))

        # set model attributes for quantized model
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.POT
        output_model.optimization_methods = [OptimizationMethod.QUANTIZATION]
        output_model.precision = [ModelPrecision.INT8]

        self.model = output_model
        self.inferencer = self.load_inferencer()
