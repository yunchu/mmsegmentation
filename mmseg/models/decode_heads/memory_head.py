# Copyright (c) 2021 CharlesPikachu
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The original repo: https://github.com/CharlesPikachu/mcibi

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from ..builder import HEADS
from ..utils import SelfAttentionBlock
from .cascade_decode_head import BaseCascadeDecodeHead


class FeaturesMemory(nn.Module):
    def __init__(self, num_classes, feats_channels, transform_channels, out_channels, num_feats_per_cls=1,
                 out_act_cfg='default', conv_cfg=None, norm_cfg=None, act_cfg=dict(type='ReLU'),
                 ignore_index=255, strategy='cosine_similarity', momentum=0.9):
        super(FeaturesMemory, self).__init__()

        assert num_feats_per_cls > 0, 'num_feats_per_cls should be larger than 0'

        # set attributes
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.transform_channels = transform_channels
        self.out_channels = out_channels
        self.num_feats_per_cls = num_feats_per_cls
        self.ignore_index = ignore_index
        self.momentum = momentum
        self.strategy = strategy
        assert self.strategy in ['mean', 'cosine_similarity']

        # init memory
        self.memory = nn.Parameter(
            torch.zeros(num_classes, num_feats_per_cls, feats_channels, dtype=torch.float),
            requires_grad=False
        )

        # define self_attention module
        if self.num_feats_per_cls > 1:
            self.self_attentions = nn.ModuleList()
            for _ in range(self.num_feats_per_cls):
                self.self_attentions.append(SelfAttentionBlock(
                    key_in_channels=feats_channels,
                    query_in_channels=feats_channels,
                    channels=transform_channels,
                    out_channels=feats_channels,
                    share_key_query=False,
                    query_downsample=None,
                    key_downsample=None,
                    key_query_num_convs=2,
                    value_out_num_convs=1,
                    key_query_norm=True,
                    value_out_norm=True,
                    matmul_norm=True,
                    with_out=True,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ))
            self.fuse_memory_conv = ConvModule(
                feats_channels * self.num_feats_per_cls,
                feats_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        else:
            self.self_attention = SelfAttentionBlock(
                key_in_channels=feats_channels,
                query_in_channels=feats_channels,
                channels=transform_channels,
                out_channels=feats_channels,
                share_key_query=False,
                query_downsample=None,
                key_downsample=None,
                key_query_num_convs=2,
                value_out_num_convs=1,
                key_query_norm=True,
                value_out_norm=True,
                matmul_norm=True,
                with_out=True,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )

        # whether need to fuse the contextual information within the input image
        self.bottleneck = ConvModule(
            2 * feats_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg if out_act_cfg == 'default' else out_act_cfg,
        )

    def forward(self, feats, prev_feats=None):
        batch_size, num_channels, h, w = feats.size()

        # extract the history features
        # --(B, num_classes, H, W) --> (B*H*W, num_classes)
        weight_cls = prev_feats.permute(0, 2, 3, 1).contiguous()
        weight_cls = weight_cls.reshape(-1, self.num_classes)
        weight_cls = F.softmax(weight_cls, dim=-1)

        # --(B*H*W, num_classes) * (num_classes, C) --> (B*H*W, C)
        selected_memory_list = []
        for idx in range(self.num_feats_per_cls):
            memory = self.memory.data[:, idx, :]
            selected_memory = torch.matmul(weight_cls, memory)
            selected_memory_list.append(selected_memory.unsqueeze(1))

        # calculate selected_memory according to the num_feats_per_cls
        if self.num_feats_per_cls > 1:
            relation_selected_memory_list = []
            for idx, selected_memory in enumerate(selected_memory_list):
                # --(B*H*W, C) --> (B, H, W, C)
                selected_memory = selected_memory.view(batch_size, h, w, num_channels)
                # --(B, H, W, C) --> (B, C, H, W)
                selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
                # --append
                relation_selected_memory_list.append(self.self_attentions[idx](feats, selected_memory))
            # --concat
            selected_memory = torch.cat(relation_selected_memory_list, dim=1)
            selected_memory = self.fuse_memory_conv(selected_memory)
        else:
            assert len(selected_memory_list) == 1
            selected_memory = selected_memory_list[0].squeeze(1)
            # --(B*H*W, C) --> (B, H, W, C)
            selected_memory = selected_memory.view(batch_size, h, w, num_channels)
            # --(B, H, W, C) --> (B, C, H, W)
            selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
            # --feed into the self attention module
            print(feats.size(), selected_memory.size())
            exit()
            selected_memory = self.self_attention(feats, selected_memory)

        memory_output = self.bottleneck(torch.cat([feats, selected_memory], dim=1))

        return memory_output

    def update(self, features, segmentation):
        batch_size, num_channels, h, w = features.size()
        momentum = self.momentum

        # use features to update memory
        segmentation = segmentation.long()
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size * h * w, num_channels)

        cls_ids = segmentation.unique()
        for cls_id in cls_ids:
            if cls_id == self.ignore_index:
                continue

            # --(B, H, W) --> (B*H*W,)
            seg_cls = segmentation.view(-1)
            # --extract the corresponding feats: (K, C)
            feats_cls = features[seg_cls == cls_id]
            # --init memory by using extracted features
            need_update = True
            for idx in range(self.num_feats_per_cls):
                if (self.memory[cls_id][idx] == 0).sum() == self.feats_channels:
                    self.memory[cls_id][idx].data.copy_(feats_cls.mean(0))
                    need_update = False
                    break
            if not need_update:
                continue

            # --update according to the selected strategy
            if self.num_feats_per_cls == 1:
                if self.strategy == 'mean':
                    feats_cls = feats_cls.mean(0)
                elif self.strategy == 'cosine_similarity':
                    similarity = F.cosine_similarity(feats_cls, self.memory[cls_id].data.expand_as(feats_cls))
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls = (feats_cls * weight.unsqueeze(-1)).sum(0)

                feats_cls = (1 - momentum) * self.memory[cls_id].data + momentum * feats_cls.unsqueeze(0)
                self.memory[cls_id].data.copy_(feats_cls)
            else:
                assert self.strategy in ['cosine_similarity']
                # ----(K, C) * (C, num_feats_per_cls) --> (K, num_feats_per_cls)
                relation = torch.matmul(
                    F.normalize(feats_cls, p=2, dim=1),
                    F.normalize(self.memory[cls_id].data.permute(1, 0).contiguous(), p=2, dim=0),
                )
                argmax = relation.argmax(dim=1)
                # ----for saving memory during training
                for idx in range(self.num_feats_per_cls):
                    mask = argmax == idx
                    feats_cls_iter = feats_cls[mask]
                    memory_cls_iter = self.memory[cls_id].data[idx].unsqueeze(0).expand_as(feats_cls_iter)
                    similarity = F.cosine_similarity(feats_cls_iter, memory_cls_iter)
                    weight = (1 - similarity) / (1 - similarity).sum()

                    feats_cls_iter = (feats_cls_iter * weight.unsqueeze(-1)).sum(0)
                    feats_cls_iter = self.memory[cls_id].data[idx] * (1 - momentum) + feats_cls_iter * momentum
                    self.memory[cls_id].data[idx].copy_(feats_cls_iter)

        # syn the memory
        if dist.is_available() and dist.is_initialized():
            memory = self.memory.data.clone()
            dist.all_reduce(memory.div_(dist.get_world_size()))
            self.memory = nn.Parameter(memory, requires_grad=False)


@HEADS.register_module()
class MemoryHead(BaseCascadeDecodeHead):
    """Mining Contextual Information Beyond Image for Semantic Segmentation.

    This head is the implementation of `MemoryNet <https://arxiv.org/abs/2108.11819>`_.
    """

    def __init__(self, transform_channels, num_feats_per_cls=1, out_act_cfg='default', sep_conv=False, **kwargs):
        super(MemoryHead, self).__init__(**kwargs)

        self.transform_channels = transform_channels
        self.num_feats_per_cls = num_feats_per_cls
        self.out_act_cfg = out_act_cfg

        self.bottleneck = self._build_conv_module(
            sep_conv,
            self.in_channels,
            self.channels,
            kernel_size=3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.memory_module = FeaturesMemory(
            num_classes=self.num_classes,
            feats_channels=self.channels,
            transform_channels=self.transform_channels,
            num_feats_per_cls=self.num_feats_per_cls,
            out_channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            out_act_cfg=self.out_act_cfg
        )

    @staticmethod
    def _build_conv_module(sep_conv, in_channels, out_channels, **kwargs):
        if sep_conv:
            return DepthwiseSeparableConvModule(
                in_channels,
                out_channels,
                dw_act_cfg=None,
                **kwargs
            )
        else:
            return ConvModule(
                in_channels,
                out_channels,
                **kwargs
            )

    def forward(self, inputs, prev_output, return_memory_input=False):
        """Forward function."""

        x = self._transform_inputs(inputs)

        memory_input = self.bottleneck(x)
        memory_output = self.memory_module(memory_input, prev_output)

        output = self.cls_seg(memory_output)

        if return_memory_input:
            return memory_input, output
        else:
            return output

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg, pixel_weights=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.
            pixel_weights (Tensor): Pixels weights.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        memory_input, seg_logits = self.forward(inputs, prev_output, return_memory_input=True)
        losses = self.losses(seg_logits, gt_semantic_seg, train_cfg, pixel_weights)

        with torch.no_grad():
            scaled_memoryInput = F.interpolate(
                memory_input,
                size=gt_semantic_seg.shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners
            )

            self.memory_module.update(
                features=scaled_memoryInput,
                segmentation=gt_semantic_seg
            )

        return losses
