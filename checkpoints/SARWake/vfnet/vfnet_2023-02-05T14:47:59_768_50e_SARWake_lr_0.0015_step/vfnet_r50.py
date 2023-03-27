dataset_type = 'CocoDataset'
data_root = 'data/coco/'
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
        classes=('Wake', ),
        ann_file='data/DATASETS/SARWake/annotations/instances_train2017.json',
        img_prefix='data/DATASETS/SARWake/train2017/',
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
        classes=('Wake', ),
        ann_file='data/DATASETS/SARWake/annotations/instances_val2017.json',
        img_prefix='data/DATASETS/SARWake/val2017/',
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
        classes=('Wake', ),
        ann_file='data/DATASETS/SARWake/annotations/instances_val2017.json',
        img_prefix='data/DATASETS/SARWake/val2017/',
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
optimizer = dict(type='SGD', lr=0.0015, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[33, 44])
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
    type='VFNet',
    backbone=dict(
        with_cp=True,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
work_dir = 'checkpoints/SARWake/vfnet/vfnet_2023-02-05T14:47:59_768_50e_SARWake_lr_0.0015_step'
auto_resume = True
gpu_ids = range(0, 1)
