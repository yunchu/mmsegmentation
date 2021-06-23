# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ShelfNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            depth=18
        )
    ),
    decode_head=dict(
        type='ShelfHead',
        in_channels=64,
        in_index=0,
        channels=64,
        input_transform=None,
        dropout_ratio=None,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
