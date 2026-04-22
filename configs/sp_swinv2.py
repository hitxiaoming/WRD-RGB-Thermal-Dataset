BATCH_SIZE = 32
CLASS_NUMS = 8
LOG_INTERVAL = 10
MAX_EPOCH = 150
SAVE_INTERVAL = 200
TRAIN_NUM_WORKERS = 8
TR_DATA_ROOT = 'data\\WRD200K8V1\\rgb\\train'
VAL_DATA_ROOT = 'data\\WRD200K8V1\\rgb\\val'
VAL_INTERVAL = 20
VAL_NUM_WORKERS = 4
combined_mean = [
    123.675,
    116.28,
    103.53,
    55.53,
    55.53,
    55.53,
]
combined_std = [
    58.395,
    57.12,
    57.375,
    14.95,
    14.95,
    14.95,
]
custom_hooks = [
    dict(interval=1, type='InteractionWeightHook'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'dual_stream_modules5m_swinv2',
    ])
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
        55.53,
        55.53,
        55.53,
    ],
    num_classes=8,
    std=[
        58.395,
        57.12,
        57.375,
        14.95,
        14.95,
        14.95,
    ],
    to_rgb=False,
    type='ClsDataPreprocessor')
default_hooks = dict(
    checkpoint=dict(interval=200, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
ir_mean = [
    55.53,
    55.53,
    55.53,
]
ir_std = [
    14.95,
    14.95,
    14.95,
]
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='tiny',
        drop_path_rate=0.1,
        fix_weights=False,
        gap_before_final_norm=True,
        in_channels=3,
        input_mode='rgb_ir',
        ir2rgb_weights=[
            0.1,
            0.1,
            0.1,
            0.1,
        ],
        out_indices=(3, ),
        rgb2ir_weights=[
            0.1,
            0.1,
            0.1,
            0.1,
        ],
        symmetric_interaction=False,
        type='DyRoadNet'),
    head=dict(
        in_channels=1536,
        loss=dict(label_smooth_val=0.1, type='LabelSmoothLoss'),
        num_classes=8,
        type='LinearClsHead'),
    neck=None,
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05))
param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        start_factor=0.001,
        type='LinearLR'),
    dict(begin=20, by_epoch=True, eta_min=1e-06, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
rgb_mean = [
    123.675,
    116.28,
    103.53,
]
rgb_std = [
    58.395,
    57.12,
    57.375,
]
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data\\WRD200K8V1\\rgb\\val',
        pipeline=[
            dict(type='LoadRGBIRCombined'),
            dict(
                backend='cv2',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=256, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='RGBIRPairDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(topk=(1, ), type='Accuracy'),
    dict(items=[
        'precision',
        'recall',
    ], type='SingleLabelMetric'),
]
test_pipeline = [
    dict(type='LoadRGBIRCombined'),
    dict(
        backend='cv2',
        edge='short',
        interpolation='bicubic',
        scale=256,
        type='ResizeEdge'),
    dict(crop_size=256, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=20)
train_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data\\WRD200K8V1\\rgb\\train',
        pipeline=[
            dict(type='LoadRGBIRCombined'),
            dict(
                backend='cv2',
                interpolation='bicubic',
                scale=256,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                erase_prob=0.25,
                fill_color=[
                    123.675,
                    116.28,
                    103.53,
                    55.53,
                    55.53,
                    55.53,
                ],
                fill_std=[
                    58.395,
                    57.12,
                    57.375,
                    14.95,
                    14.95,
                    14.95,
                ],
                max_area_ratio=0.3333333333333333,
                min_area_ratio=0.02,
                mode='rand',
                type='RandomErasingMultiChannel'),
            dict(type='PackInputs'),
        ],
        type='RGBIRPairDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadRGBIRCombined'),
    dict(
        backend='cv2',
        interpolation='bicubic',
        scale=256,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        erase_prob=0.25,
        fill_color=[
            123.675,
            116.28,
            103.53,
            55.53,
            55.53,
            55.53,
        ],
        fill_std=[
            58.395,
            57.12,
            57.375,
            14.95,
            14.95,
            14.95,
        ],
        max_area_ratio=0.3333333333333333,
        min_area_ratio=0.02,
        mode='rand',
        type='RandomErasingMultiChannel'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data\\WRD200K8V1\\rgb\\val',
        pipeline=[
            dict(type='LoadRGBIRCombined'),
            dict(
                backend='cv2',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=256, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='RGBIRPairDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(topk=(1, ), type='Accuracy'),
    dict(items=[
        'precision',
        'recall',
    ], type='SingleLabelMetric'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'fastroadnetplus/WRD200K8V1/v5_rid_1111_11m_150'
