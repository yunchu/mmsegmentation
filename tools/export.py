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
import numpy as np
from mmcv import DictAction
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint
from torch import nn

from mmseg.models import build_segmentor

torch.manual_seed(3)


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


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    segs = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)
    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
    } for _ in range(N)]
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }
    return mm_inputs


def _update_input_img(img_list, img_meta_list, update_ori_shape=False):
    # update img and its meta list
    N, C, H, W = img_list[0].shape
    img_meta = img_meta_list[0][0]
    img_shape = (H, W, C)
    if update_ori_shape:
        ori_shape = img_shape
    else:
        ori_shape = img_meta['ori_shape']
    pad_shape = img_shape
    new_img_meta_list = [[{
        'img_shape':
        img_shape,
        'ori_shape':
        ori_shape,
        'pad_shape':
        pad_shape,
        'filename':
        img_meta['filename'],
        'scale_factor':
        (img_shape[1] / ori_shape[1], img_shape[0] / ori_shape[0]) * 2,
        'flip':
        False,
    } for _ in range(N)]]

    return img_list, new_img_meta_list


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


def export_to_openvino(cfg, onnx_model_path, output_dir_path, input_shape=None,
                       input_format='rgb', precision='FP32'):
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

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
                   f'--output="{output_names}"' \
                   f'--data_type {precision}'

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

    img_scale = cfg.test_pipeline[1]['img_scale']
    input_shape = (1, 3, img_scale[1], img_scale[0])

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
        export_to_openvino(
            cfg,
            onnx_model_path,
            args.output_dir,
            input_shape,
            args.input_format
        )


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
