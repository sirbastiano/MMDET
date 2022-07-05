# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.01,
    step=[30, 40, 50, 60, 70, 80])
runner = dict(type='EpochBasedRunner', max_epochs=100)
