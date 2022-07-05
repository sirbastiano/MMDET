home_path = '/home/sirbastiano/Documenti/Scripts/MMDETv2/mmdetection/configs/Baseline/'

_base_ = [
    # '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/wake_detection.py',
    '_base_/schedules/schedule_40e.py', 
    '_base_/default_runtime.py',
]

_base_ = [home_path+x for x in _base_]


model = dict(
    type='CornerNet',
    backbone=dict(
        in_channels=1,
        type='HourglassNet',
        downsample_times=5,
        num_stacks=2,
        stage_channels=[256, 256, 384, 384, 384, 512],
        stage_blocks=[2, 2, 2, 2, 2, 4],
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=None,
    bbox_head=dict(
        type='CentripetalHead',
        num_classes=1,
        in_channels=256,
        num_feat_levels=2,
        corner_emb_channels=0,
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1),
        loss_guiding_shift=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=0.05),
        loss_centripetal_shift=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=1)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        corner_topk=100,
        local_maximum_kernel=3,
        distance_threshold=0.5,
        score_thr=0.05,
        max_per_img=100,
        nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian')))

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_size = (511, 511)


# img_norm_cfg = dict(
#     mean=[128, 128, 128], std=[70, 70, 70], to_rgb=True)

img_norm_cfg = dict(
    mean=128, std=70, to_rgb=True)

classes = ('wake',)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]



# test_pipeline = [
#     dict(type='LoadImageFromFile',to_float32=True),
#     dict(type='Resize', img_scale=img_size, keep_ratio=False),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
#                            'scale_factor', 'flip', 'img_norm_cfg')),
# ]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=False),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=True,
        transforms=[
            dict(type='Resize',img_scale=img_size, keep_ratio=False),
            dict(
                type='RandomCenterCropPad',
                crop_size=None,
                ratios=None,
                border=None,
                test_mode=True,
                test_pad_mode=['logical_or', 127],
                **img_norm_cfg),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                        #    'scale_factor', 'flip', 'img_norm_cfg', 'border')),
                           'scale_factor', 'flip', 'img_norm_cfg')),
        ])
]





data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

work_dir = 'checkpoints/Centripetalnet_40e'  # Directory to save the model checkpoints and logs for the current experiments.
