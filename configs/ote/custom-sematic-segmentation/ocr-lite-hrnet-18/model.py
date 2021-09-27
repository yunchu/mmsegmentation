_base_ = [
    './data_pipeline.py'
]

# # pre-trained params settings
# ignore_keys = [r'^backbone\.increase_modules\.', r'^backbone\.increase_modules\.',
#                r'^backbone\.downsample_modules\.', r'^backbone\.final_layer\.',
#                r'^head\.']

# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
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
                extra_stride=False,
                input_norm=False,
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
                    channels=320
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
                enable=True
            ),
            add_input=False
        )
    ),
    decode_head=[
        dict(type='FCNHead',
             in_channels=40,
             in_index=0,
             channels=40,
             input_transform=None,
             kernel_size=1,
             num_convs=0,
             concat_input=False,
             dropout_ratio=-1,
             num_classes=2,
             norm_cfg=norm_cfg,
             align_corners=False,
             enable_out_norm=False,
             loss_decode=[
                 dict(type='CrossEntropyLoss',
                      use_sigmoid=False,
                      loss_jitter_prob=0.01,
                      sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                      loss_weight=1.0),
             ]),
        dict(type='OCRHead',
             in_channels=40,
             in_index=0,
             channels=40,
             ocr_channels=40,
             sep_conv=True,
             input_transform=None,
             dropout_ratio=-1,
             num_classes=2,
             norm_cfg=norm_cfg,
             align_corners=False,
             enable_out_norm=True,
             loss_decode=[
                 dict(type='AMSoftmaxLoss',
                      scale_cfg=dict(
                          type='PolyScalarScheduler',
                          start_scale=30,
                          end_scale=5,
                          num_iters=30000,
                          power=1.2
                      ),
                      margin_type='cos',
                      margin=0.5,
                      gamma=0.0,
                      t=1.0,
                      target_loss='ce',
                      pr_product=False,
                      conf_penalty_weight=dict(
                          type='PolyScalarScheduler',
                          start_scale=0.2,
                          end_scale=0.15,
                          num_iters=20000,
                          power=1.2
                      ),
                      loss_jitter_prob=0.01,
                      border_reweighting=False,
                      sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
                      loss_weight=1.0),
             ]),
    ],
    train_cfg=dict(
        mix_loss=dict(enable=False, weight=0.1)
    ),
    test_cfg=dict(
        mode='whole'
    ),
)

find_unused_parameters = True

# optimizer
optimizer = dict(
    type='SGD',
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.0005
)
optimizer_config = dict(
    grad_clip=dict(
        method='default',
        max_norm=40,
        norm_type=2
    )
)

# parameter manager
params_config = dict(
    type='FreezeLayers',
    by_epoch=False,
    iters=2000,
    open_layers=[r'backbone\.aggregator\.', r'neck\.', r'decode_head\.', r'auxiliary_head\.']
)

# learning policy
lr_config = dict(
    policy='customstep',
    by_epoch=False,
    gamma=0.1,
    step=[20000, 30000],
    fixed='constant',
    fixed_iters=2000,
    fixed_ratio=10.0,
    warmup='cos',
    warmup_iters=4000,
    warmup_ratio=1e-2,
)

# runtime settings
runner = dict(
    type='IterBasedRunner',
    max_iters=40000
)
checkpoint_config = dict(
    by_epoch=False,
    interval=1000
)
evaluation = dict(
    interval=1000,
    metric='mIoU'
)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/media/experiments/segmentation/kvasir_seg/v77/iter_21000.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
