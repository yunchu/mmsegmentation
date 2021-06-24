# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='EfficientNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            model_name='efficientnet-b1',
            out_feat_scale=(1., 0.166, 0.2, 0.25, 0.4),
            add_input=True
        )
    ),
    decode_head=dict(
        type='HyperSegHead',
        in_channels=(3, 16, 4, 8, 28, 128, 1280),
        in_index=(0, 1, 2, 3, 4, 5, 6),
        channels=None,
        weight_levels=2,
        kernel_sizes=[1, 1, 1, 3, 3],
        level_channels=[32, 16, 8, 8, 8],
        expand_ratio=2,
        with_out_fc=False,
        decoder_dropout=None,
        weight_groups=[32, 16, 8, 16, 4],
        decoder_groups=1,
        unify_level=4,
        coords_res=[(768, 768), (768, 1536)],
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
