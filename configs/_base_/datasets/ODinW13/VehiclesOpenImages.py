# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/odinw/'
CLASS_NAMES = ('object',)
data_root = data_root +  'VehiclesOpenImages/416x416/'
VAL_ANN   = 'valid/annotations_one_class.json'  # change: 验证标注文件

backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=VAL_ANN,
        data_prefix=dict(img='valid/'),
        test_mode=True,
        pipeline=test_pipeline,
        return_classes=True,
        backend_args=backend_args,
        metainfo=dict(classes=CLASS_NAMES)))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + VAL_ANN,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator
