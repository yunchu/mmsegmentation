_base_ = [
    '../_base_/models/fast_scnn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# Re-config the model head.
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)
model = dict(
    decode_head=dict(num_classes=150),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=32,
            num_convs=1,
            num_classes=150,
            in_index=-2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=32,
            num_convs=1,
            num_classes=150,
            in_index=-3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    ])

# Re-config the data sampler.
data = dict(samples_per_gpu=2, workers_per_gpu=4)

# Re-config the optimizer.
optimizer = dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=4e-5)
