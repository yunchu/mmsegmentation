# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from torch import nn

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class CascadeEncoderDecoder(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        self.num_stages = num_stages
        super(CascadeEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""

        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head.append(builder.build_head(decode_head[i]))
        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes

    def set_step_params(self, init_iter, epoch_size):
        for decode_head in self.decode_head:
            decode_head.set_step_params(init_iter, epoch_size)

        if self.auxiliary_head is not None:
            if isinstance(self.auxiliary_head, list):
                for aux_head in self.auxiliary_head:
                    aux_head.set_step_params(init_iter, epoch_size)
            else:
                self.auxiliary_head.set_step_params(init_iter, epoch_size)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        self.backbone.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            self.decode_head[i].init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        x = self.extract_feat(img)

        out = self.decode_head[0].forward_test(x, img_metas, self.test_cfg)
        for i in range(1, self.num_stages):
            out = self.decode_head[i].forward_test(x, out, img_metas, self.test_cfg)

        out_scale = self.test_cfg.get('output_scale', None)
        if out_scale is not None and not self.training:
            out = out_scale * out

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )

        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, pixel_weights=None):
        """Run forward function and calculate loss for decode head in
        training."""

        losses = dict()

        loss_decode = self.decode_head[0].forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg, pixel_weights
        )
        losses.update(add_prefix(loss_decode, 'decode_0'))

        for i in range(1, self.num_stages):
            prev_scale = self.decode_head[i - 1].last_scale
            prev_logits = self.decode_head[i - 1].forward_test(x, img_metas, self.test_cfg)

            prev_scaled_logits = prev_scale * prev_logits
            loss_decode = self.decode_head[i].forward_train(
                x, prev_scaled_logits, img_metas, gt_semantic_seg, self.train_cfg, pixel_weights
            )
            losses.update(add_prefix(loss_decode, f'decode_{i}'))

        return losses
