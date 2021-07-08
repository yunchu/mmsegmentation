_base_ = [
    '../_base_/models/stdcnet-813.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_cos_160k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        type='STDCNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            backbone='STDCNet1446'
        )
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=-1,
        channels=256,
        input_transform=None,
        kernel_size=1,
        num_convs=1,
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
