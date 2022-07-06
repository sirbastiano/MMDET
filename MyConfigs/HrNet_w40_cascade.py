_base_ = './HrNet_w32_cascade.py'
# model settings
model = dict(
    pretrained=None,
    init_cfg=None,
    backbone=dict(
        norm_eval=False,
        init_cfg=None,  
        type='HRNet',
        extra=dict(
            stage2=dict(num_channels=(40, 80)),
            stage3=dict(num_channels=(40, 80, 160)),
            stage4=dict(num_channels=(40, 80, 160, 320))),
        # init_cfg=dict(
        #     type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w40')
            ),
    neck=dict(type='HRFPN', in_channels=[40, 80, 160, 320], out_channels=256))



# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[3, 8, 12, 16])
runner = dict(type='EpochBasedRunner', max_epochs=20)