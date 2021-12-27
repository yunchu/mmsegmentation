# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import sys

import torch
import torch._C
import torch.serialization
import mmcv
from mmcv import DictAction
from mmcv.runner import load_checkpoint

from mmseg.apis import export_model
from mmseg.models import build_segmentor


torch.manual_seed(42)


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked

    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))

    del module

    return module_output


def parse_args():
    parser = argparse.ArgumentParser(description='Export the MMSeg model to ONNX/IR')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help="path to file with model's weights")
    parser.add_argument('output_dir', help='path to directory to save exported models in')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='Override some settings in the used config, the key-value pair '
                             'in xxx=yyy format will be merged into config file. If the value to '
                             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                             'Note that the quotation marks are necessary and that no white space '
                             'is allowed.')

    subparsers = parser.add_subparsers(title='target', dest='target', help='target model format')
    subparsers.required = True
    parser_onnx = subparsers.add_parser('onnx', help='export to ONNX')
    parser_openvino = subparsers.add_parser('openvino', help='export to OpenVINO')
    parser_openvino.add_argument('--input-format', choices=['BGR', 'RGB'], default='BGR',
                                 help='Input image format for exported model.')

    return parser.parse_args()


def main(args):
    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    segmentor = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    # convert SyncBN to BN
    segmentor = _convert_batchnorm(segmentor)

    if args.checkpoint:
        load_checkpoint(segmentor, args.checkpoint, map_location='cpu')
    
    export_model(segmentor,
                 cfg,
                 args.output_dir,
                 target=args.target,
                 onnx_opset=args.opset,
                 output_logits=True,
                 input_format=args.input_format)


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
