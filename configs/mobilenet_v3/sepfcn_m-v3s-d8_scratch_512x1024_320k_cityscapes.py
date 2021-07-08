_base_ = './lraspp_m-v3-d8_scratch_512x1024_320k_cityscapes.py'

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MobileNetV3',
        arch='small',
        out_indices=(12,),
        norm_cfg=norm_cfg
    ),
    decode_head=dict(
        type='DepthwiseSeparableFCNHead',
        in_channels=576,
        channels=128,
        num_convs=1,
        concat_input=False,
        num_classes=19,
        in_index=-1,
        norm_cfg=norm_cfg,
        align_corners=False,
        input_transform=None,
        sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
)
