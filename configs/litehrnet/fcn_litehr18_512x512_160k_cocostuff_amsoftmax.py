_base_ = [
    '../_base_/models/fcn_litehr18.py', '../_base_/datasets/coco_stuff.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_step_160k_ml.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='FCNHead',
        in_channels=[40, 40, 80, 160],
        in_index=[0, 1, 2, 3],
        channels=320,
        input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=181,
        norm_cfg=norm_cfg,
        align_corners=False,
        enable_out_norm=True,
        loss_decode=[
            dict(type='AMSoftmaxLoss',
                 scale_cfg=dict(
                     type='PolyScalarScheduler',
                     start_scale=30,
                     end_scale=5,
                     num_iters=130000,
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
                     num_iters=100000,
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
    interval=4000,
    metric='mIoU'
)

find_unused_parameters = True
