_base_ = [
    '../_base_/models/hyperseg_litehr18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_cos_160k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
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
                    enable=True,
                    channels=576
                ),
                position_att=dict(
                    enable=True,
                    key_channels=128,
                    value_channels=320,
                    psp_size=(1, 3, 6, 8),
                ),
                local_att=dict(
                    enable=True
                )
            ),
            out_aggregator=dict(
                enable=False
            ),
            add_input=True
        )
    )
)
