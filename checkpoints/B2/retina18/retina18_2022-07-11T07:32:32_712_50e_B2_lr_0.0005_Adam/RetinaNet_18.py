dataset_type = 'CocoDataset'
data_root = 'data/DATASETS/B2'
img_size = [(768, 768)]
classes = ('wake', )
img_norm_cfg = dict(mean=[128, 128, 128], std=[70, 70, 70], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(768, 768)], keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='diagonal'),
    dict(
        type='Normalize', mean=[128, 128, 128], std=[70, 70, 70], to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(type='Resize', img_scale=[(768, 768)], keep_ratio=False),
    dict(
        type='Normalize', mean=[128, 128, 128], std=[70, 70, 70], to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        meta_keys=('filename', 'ori_shape', 'img_shape', 'scale_factor'),
        keys=['img']),
    dict(type='WrapFieldsToLists')
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        classes=('wake', ),
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile', color_type='color'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=[(768, 768)], keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
            dict(type='RandomFlip', flip_ratio=0.5, direction='diagonal'),
            dict(
                type='Normalize',
                mean=[128, 128, 128],
                std=[70, 70, 70],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=('wake', ),
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile', color_type='color'),
            dict(type='Resize', img_scale=[(768, 768)], keep_ratio=False),
            dict(
                type='Normalize',
                mean=[128, 128, 128],
                std=[70, 70, 70],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'scale_factor'),
                keys=['img']),
            dict(type='WrapFieldsToLists')
        ]),
    test=dict(
        type='CocoDataset',
        classes=('wake', ),
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile', color_type='color'),
            dict(type='Resize', img_scale=[(768, 768)], keep_ratio=False),
            dict(
                type='Normalize',
                mean=[128, 128, 128],
                std=[70, 70, 70],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'scale_factor'),
                keys=['img']),
            dict(type='WrapFieldsToLists')
        ]))
evaluation = dict(interval=1000, metric='bbox')
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[30, 38])
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=250,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
opencv_num_threads = 1
mp_start_method = 'spawn'
home = '/home/sirbastiano/Documenti/Scripts/MMDETv2/mmdetection/configs/'
model = dict(
    type='RetinaNet',
    backbone=dict(
        in_channels=3,
        with_cp=True,
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
work_dir = 'checkpoints/B2/retina18/retina18_2022-07-11T07:32:32_712_50e_B2_lr_0.0005_Adam'
auto_resume = True
gpu_ids = range(0, 1)
