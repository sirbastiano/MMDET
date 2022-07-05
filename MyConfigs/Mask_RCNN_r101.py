
_base_ = './Mask_RCNN_r50.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))


work_dir = 'checkpoints/Mask_RCNN_r101_FPN_default'  # Directory to save the model checkpoints and logs for the current experiments.

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[25, 35])
runner = dict(type='EpochBasedRunner', max_epochs=40)