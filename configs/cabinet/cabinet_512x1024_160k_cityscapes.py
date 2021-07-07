_base_ = [
    '../_base_/models/cabinet.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_cos_160k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='DepthwiseSeparableFCNHead',
        in_channels=128,
        channels=128,
        concat_input=False,
        num_classes=19,
        in_index=-1,
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
