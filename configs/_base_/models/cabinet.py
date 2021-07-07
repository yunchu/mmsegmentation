# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='CABiNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        in_channels=3,
        extra=dict()
    ),
    decode_head=dict(
        type='DepthwiseSeparableFCNHead',
        in_channels=128,
        channels=128,
        concat_input=False,
        num_classes=19,
        in_index=-1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=20
        )
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
