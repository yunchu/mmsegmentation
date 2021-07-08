_base_ = [
    '../_base_/models/fcn_litehr18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_cos_160k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        type='LiteHRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stem=dict(
                stem_channels=32,
                out_channels=32,
                expand_ratio=1,
                extra_stride=True
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
            out_aggregator=dict(
                enable=True
            )
        )
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=40,
        in_index=0,
        channels=40,
        input_transform=None,
        kernel_size=1,
        num_convs=0,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
)
