# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

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

import sys
import argparse

import mmcv
import torch
from openvino.inference_engine import IECore  # pylint: disable=no-name-in-module
from mmcv.utils import DictAction

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.core.utils import propagate_root_dir


def update_config(cfg):
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
    cfg.data.test.pipeline[1].transforms[2] = dict(type='Normalize', **img_norm_cfg)

    return cfg


def collect_results(model, data_loader):
    results = []
    progress_bar = mmcv.ProgressBar(len(data_loader.dataset))

    for data in data_loader:
        input_data = data['img'][0].cpu().numpy()
        result = model(input_data)

        results.extend(result.squeeze(axis=0))

        batch_size = len(input_data)
        for _ in range(batch_size):
            progress_bar.update()

    return results


def load_ie_core(device='CPU', cpu_extension=None):
    ie = IECore()
    if device == 'CPU' and cpu_extension:
        ie.add_extension(cpu_extension, 'CPU')

    return ie


class IEModel:
    def __init__(self, model_path, ie_core, device='CPU', num_requests=1):
        if model_path.endswith((".xml", ".bin")):
            model_path = model_path[:-4]
        self.net = ie_core.read_network(model_path + ".xml", model_path + ".bin")
        assert len(self.net.input_info) == 1, "One input is expected"

        self.exec_net = ie_core.load_network(
            network=self.net, device_name=device, num_requests=num_requests
        )

        self.input_name = next(iter(self.net.input_info))
        if len(self.net.outputs) > 1:
            raise Exception("One output is expected")
        else:
            self.output_name = next(iter(self.net.outputs))

        self.input_size = self.net.input_info[self.input_name].input_data.shape
        self.output_size = self.exec_net.requests[0].output_blobs[self.output_name].buffer.shape
        self.num_requests = num_requests

    def infer(self, data):
        input_data = {self.input_name: data}
        infer_result = self.exec_net.infer(input_data)

        return infer_result[self.output_name]


class Segmentor(IEModel):
    def __init__(self, model_path, ie_core, device='CPU', num_requests=1):
        super().__init__(model_path, ie_core, device, num_requests)

    def __call__(self, input_data):
        return self.infer(input_data)


def main(args):
    assert args.model.endswith('.xml')

    # load config
    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_config(cfg)
    cfg = propagate_root_dir(cfg, args.data_dir)

    # build the dataset
    dataset = build_dataset(cfg.data.test)

    # build the dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # load model
    ie_core = load_ie_core()
    model = Segmentor(args.model, ie_core)

    # collect results
    outputs = collect_results(model, data_loader)

    # get metrics
    eval_kwargs = {} if args.eval_options is None else args.eval_options
    if args.eval:
        dataset.evaluate(outputs, args.eval, show_log=True, **eval_kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description='Test model deployed to ONNX or OpenVINO')
    parser.add_argument('config', help='path to configuration file')
    parser.add_argument('model', help='path to onnx model file or xml file in case of OpenVINO.')
    parser.add_argument('--data_dir', type=str,
                        help='the dir with dataset')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
                             ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--options', nargs='+', action=DictAction,
                        help='custom options')
    parser.add_argument('--eval-options', nargs='+', action=DictAction,
                        help='custom options for evaluation')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
