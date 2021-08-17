import argparse
import sys
from functools import partial
from os import makedirs
from os.path import exists, dirname, basename, splitext, join
from subprocess import run, CalledProcessError, DEVNULL

import onnx
import torch
import torch._C
import torch.serialization
import mmcv
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint
from torch import nn

from mmseg.models import build_segmentor
from .pytorch2onnx import _convert_batchnorm, _demo_mm_inputs, _update_input_img

torch.manual_seed(3)


def pytorch2onnx(model,
                 mm_inputs,
                 opset_version=11,
                 output_file='tmp.onnx',
                 verify=True,
                 dynamic_export=True):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        mm_inputs (dict): Contain the input tensors and img_metas information.
        opset_version (int): The onnx op version. Default: 11.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether run ONNX check.
            Default: False.
        dynamic_export (bool): Whether to export ONNX with dynamic axis.
            Default: False.
    """

    model.cpu().eval()

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    img_list = [img[None, :] for img in imgs]
    img_meta_list = [[img_meta] for img_meta in img_metas]
    # update img_meta
    img_list, img_meta_list = _update_input_img(img_list, img_meta_list)

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(
        model.forward,
        img_metas=img_meta_list,
        return_loss=False,
        rescale=True
    )

    dynamic_axes = None
    if dynamic_export:
        if model.test_cfg.mode == 'slide':
            dynamic_axes = {'input': {0: 'batch'}, 'output': {1: 'batch'}}
        else:
            dynamic_axes = {
                'input': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'
                },
                'output': {
                    1: 'batch',
                    2: 'height',
                    3: 'width'
                }
            }

    register_extra_symbolics(opset_version)

    with torch.no_grad():
        torch.onnx.export(
            model, (img_list, ),
            output_file,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=False,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes
        )
        print(f'Successfully exported ONNX model: {output_file}')

    model.forward = origin_forward

    if verify:
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)


def export_to_openvino(cfg, onnx_model_path, output_dir_path, input_shape=None, input_format='rgb'):
    onnx_model = onnx.load(onnx_model_path)

    output_names = set(out.name for out in onnx_model.graph.output)
    # Clear names of the nodes that produce network's output blobs.
    for node in onnx_model.graph.node:
        if output_names.intersection(node.output):
            node.ClearField('name')
    onnx.save(onnx_model, onnx_model_path)
    output_names = ','.join(output_names)

    normalize = [v for v in cfg.data.test.pipeline[1].transforms if v['type'] == 'Normalize'][0]

    mean_values = normalize['mean']
    scale_values = normalize['std']
    command_line = f'mo.py --input_model="{onnx_model_path}" ' \
                   f'--mean_values="{mean_values}" ' \
                   f'--scale_values="{scale_values}" ' \
                   f'--output_dir="{output_dir_path}" ' \
                   f'--output="{output_names}"'

    assert input_format.lower() in ['bgr', 'rgb']

    if input_shape is not None:
        command_line += f' --input_shape="{input_shape}"'
    if normalize['to_rgb'] and input_format.lower() == 'bgr' or \
       not normalize['to_rgb'] and input_format.lower() == 'rgb':
        command_line += ' --reverse_input_channels'

    try:
        run('mo.py -h', stdout=DEVNULL, stderr=DEVNULL, shell=True, check=True)
    except CalledProcessError:
        print('OpenVINO Model Optimizer not found, please source '
              'openvino/bin/setupvars.sh before running this script.')
        return

    print(command_line)
    run(command_line, shell=True, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Export the MMSeg model to ONNX/IR')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help="path to file with model's weights")
    parser.add_argument('output-dir', help='path to directory to save exported models in')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset')

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

    if args.shape is None:
        img_scale = cfg.test_pipeline[1]['img_scale']
        input_shape = (1, 3, img_scale[1], img_scale[0])
    elif len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    segmentor = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    # convert SyncBN to BN
    segmentor = _convert_batchnorm(segmentor)

    if args.checkpoint:
        checkpoint = load_checkpoint(
            segmentor, args.checkpoint, map_location='cpu')
        segmentor.CLASSES = checkpoint['meta']['CLASSES']
        segmentor.PALETTE = checkpoint['meta']['PALETTE']

    # create dummy input
    if isinstance(segmentor.decode_head, nn.ModuleList):
        num_classes = segmentor.decode_head[-1].num_classes
    else:
        num_classes = segmentor.decode_head.num_classes
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    onnx_model_path = join(args.output_dir, splitext(basename(args.config))[0] + '.onnx')
    base_output_dir = dirname(onnx_model_path)
    if not exists(base_output_dir):
        makedirs(base_output_dir)

    # convert model to onnx file
    pytorch2onnx(
        segmentor,
        mm_inputs,
        opset_version=args.opset,
        output_file=onnx_model_path,
    )

    if args.target == 'openvino':
        export_to_openvino(cfg, onnx_model_path, args.output_dir, input_shape, args.input_format)


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
