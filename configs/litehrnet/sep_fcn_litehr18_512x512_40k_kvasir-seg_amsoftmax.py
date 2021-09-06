_base_ = [
    '../_base_/models/fcn_litehr18.py', '../_base_/datasets/kvasir.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_cos_40k_ml.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='DepthwiseSeparableFCNHead',
        in_channels=40,
        in_index=0,
        channels=40,
        input_transform=None,
        kernel_size=3,
        num_convs=2,
        concat_input=False,
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
        ]
    ),
    train_cfg=dict(
        mix_loss=dict(enable=True, weight=0.1)
    ),
)
evaluation = dict(
    metric='mDice',
)

find_unused_parameters = True
