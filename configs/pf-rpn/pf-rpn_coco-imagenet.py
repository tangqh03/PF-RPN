_base_ = [
    './grounding_dino_swin-b_pf-rpn_base.py',
]

custom_imports = dict(
    imports=['mmdet.models.detectors.pf_rpn'],
    allow_failed_imports=False
)

data_root = 'data/coco'

model = dict(
    type='PFRPN',
    num_pseudo_tokens=3,
    sp_thr=0.3,
    sp_iter_num=3,
    topk=2,
    bbox_head=dict(
        num_classes=1,
        contrastive_cfg=dict(max_text_len=3),
        rm_lang_embed=True,
        use_conf_branch=True,
        loss_ctr=dict(type='L1Loss', loss_weight=5.0),
    ),
)

custom_classes = ('object',)
train_dataloader = dict(
    num_workers=4,
    pin_memory=True,
    dataset=dict(
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=False),
        return_classes=True,
        ann_file='annotations/merged_one_class_area.json',
        data_prefix=dict(img='train2017/'),
        metainfo=dict(classes=custom_classes)))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    pin_memory=True,
    dataset=dict(
        return_classes=True,
        data_root=data_root,
        ann_file='annotations/instances_val2017_sc.json',
        data_prefix=dict(img='val2017/'),
        metainfo=dict(classes=custom_classes)))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file='data/coco/annotations/instances_val2017_1p_sc.json')
test_evaluator = val_evaluator

optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict({
            'meta_net': dict(lr_mult=0.1),
            'conf_branches': dict(lr_mult=1.0),
            'router': dict(lr_mult=1.0),
            'learnable_text_embedding': dict(lr_mult=0.1),
            'absolute_pos_embed': dict(decay_mult=0.0),
            'backbone': dict(lr_mult=0.5),
            'backbone.channel_align_layers': dict(lr_mult=10.0),
            'decoder': dict(lr_mult=0.0),
        })),
    type='OptimWrapper')

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
    static_graph=True)

custom_hooks = [
    dict(type='SelectiveFinetuneHook'),
]

train_cfg = dict(max_epochs=1, type='EpochBasedTrainLoop', val_interval=1)

work_dir = 'work_dirs/train/coco/pf-rpn'
