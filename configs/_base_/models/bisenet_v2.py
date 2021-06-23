# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth',
    backbone=dict(
        type='BiSeNetV2',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict()
    ),
    decode_head=dict(
        type='BiSeHead',
        in_channels=128,
        in_index=-1,
        channels=1024,
        input_transform=None,
        up_factor=8,
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
