_base_ = [
    '../_base_/models/hyperseg_litehr18.py', '../_base_/datasets/kvasir.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_step_40k_ml.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='HyperSegHead',
        in_channels=(3, 40, 80, 160, 320),
        in_index=(0, 1, 2, 3, 4),
        channels=None,
        weight_levels=2,
        weight_same_last_level=True,
        kernel_sizes=[1, 3, 3, 3],
        level_channels=[32, 16, 8, 8],
        expand_ratio=1,
        with_out_fc=True,
        with_out_norm=True,
        decoder_dropout=None,
        weight_groups=[8, 8, 4, 4],
        decoder_groups=1,
        unify_level=5,
        act_cfg=dict(type='ReLU6'),
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
                     end_scale=10,
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
                     end_scale=0.085,
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
        mix_loss=dict(enable=False, weight=0.1)
    ),
)
evaluation = dict(
    metric='mDice',
)

find_unused_parameters = True
