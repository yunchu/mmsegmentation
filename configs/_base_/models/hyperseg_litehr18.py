# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='LiteHRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stem=dict(
                stem_channels=32,
                out_channels=32,
                expand_ratio=1,
                strides=(2, 2),
                extra_stride=False
            ),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                )
            ),
            out_modules=dict(
                conv=dict(
                    enable=False,
                    channels=576
                ),
                position_att=dict(
                    enable=False,
                    key_channels=128,
                    value_channels=320,
                    psp_size=(1, 3, 6, 8),
                ),
                local_att=dict(
                    enable=False
                )
            ),
            out_aggregator=dict(
                enable=False
            ),
            add_input=True
        )
    ),
    decode_head=dict(
        type='HyperSegHead',
        in_channels=(3, 32, 40, 80, 160, 320, 576),
        in_index=(0, 1, 2, 3, 4, 5, 6),
        channels=None,
        weight_levels=2,
        weight_same_last_level=True,
        kernel_sizes=[1, 1, 1, 3, 3],
        level_channels=[32, 16, 8, 8, 8],
        expand_ratio=2,
        with_out_fc=False,
        decoder_dropout=None,
        weight_groups=[32, 16, 8, 16, 4],
        decoder_groups=1,
        unify_level=4,
        num_classes=19,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU6'),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
            loss_weight=1.0
        )
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
