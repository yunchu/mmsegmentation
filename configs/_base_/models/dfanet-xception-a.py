# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='DFANet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            fc_channels=1000
        )
    ),
    decode_head=dict(
        type='DFAHead',
        in_channels=[48, 48, 48, 192, 192, 192],
        in_index=[0, 1, 2, 3, 4, 5],
        channels=None,
        input_transform='multiple_select',
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
