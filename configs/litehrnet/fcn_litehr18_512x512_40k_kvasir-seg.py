_base_ = [
    '../_base_/models/fcn_litehr18.py', '../_base_/datasets/kvasir.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_cos_40k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
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
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        sampler=dict(type='MaxPoolingPixelSampler', ratio=0.25, p=1.7),
        sampler_loss_idx=0,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # dict(type='BoundaryLoss', loss_weight=0.01)
        ]
    )
)

checkpoint_config = dict(
    by_epoch=False,
    interval=1000,
)
evaluation = dict(
    interval=1000,
    metric='mDice'
)

# parameter manager
params_config = dict(
    type='FreezeLayers',
    by_epoch=False,
    iters=0,
    open_layers=[r'backbone\.aggregator\.', r'neck\.', r'decode_head\.', r'auxiliary_head\.']
)

# optimizer
optimizer = dict(
    lr=1e-2,
)

# learning policy
lr_config = dict(
    policy='customcos',
    by_epoch=False,
    periods=[36000],
    min_lr_ratio=1e-3,
    alpha=1.2,
    # fixed='constant',
    # fixed_iters=2000,
    # fixed_ratio=10.0,
    warmup='cos',
    warmup_iters=4000,
    warmup_ratio=1e-3,
)

find_unused_parameters = True
