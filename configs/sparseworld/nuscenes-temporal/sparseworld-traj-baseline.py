"""Frozen reference configuration for the original SparseWorld SCF path."""

_base_ = ['./sparseworld-traj-finetune.py']

model = dict(
    dsqe_mode='baseline',
    dsqe_cfg=dict(enabled=False, mode='baseline'))

# Keep the reference path independent from PreSCF actor supervision.  This is
# intentionally spelled out here instead of mutating the inherited Python
# variable from ``sparseworld-traj-finetune.py``: MMCV evaluates child config
# files before merging their namespaces, so inherited local variables are not
# available for such an in-place edit.  The transforms and non-actor keys are
# otherwise identical to the original BaseLine pipeline.
_baseline_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False,
         color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=4),
    dict(type='LoadOccGTFromFile4DTraj'),
    dict(type='RandomTransformImage',
         ida_aug_conf=dict(
             resize_lim=(0.38, 0.55), final_dim=(256, 704),
             bot_pct_lim=(0.0, 0.0), rot_lim=(0.0, 0.0),
             H=900, W=1600, rand_flip=True),
         training=True),
    dict(type='DefaultFormatBundle3D', class_names=_baseline_class_names),
    dict(
        type='Collect4D',
        keys=[
            'img', 'voxel_semantics', 'mask_lidar', 'mask_camera', 'rays',
            'temporal_semantics', 'temporal_rays', 'temporal_ego_states',
            'temporal_trajs', 'temporal2ego', 'temporal_adjacent2ego',
            'temporal_ego2global'
        ],
        meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img',
            'img_timestamp', 'ego2lidar', 'ego2global', 'sample_idx'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False,
         color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=4,
         test_mode=False),
    dict(type='LoadOccGTFromFile4DTraj'),
    dict(type='RandomTransformImage',
         ida_aug_conf=dict(
             resize_lim=(0.38, 0.55), final_dim=(256, 704),
             bot_pct_lim=(0.0, 0.0), rot_lim=(0.0, 0.0),
             H=900, W=1600, rand_flip=True),
         training=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=_baseline_class_names,
                with_label=False),
            dict(
                type='Collect4D',
                keys=[
                    'img', 'voxel_semantics', 'mask_lidar', 'mask_camera',
                    'temporal_semantics', 'temporal_ego_states',
                    'temporal_ego2global', 'temporal_trajs'
                ],
                meta_keys=[
                    'filename', 'box_type_3d', 'ori_shape', 'img_shape',
                    'pad_shape', 'sample_idx', 'lidar2img', 'img_timestamp',
                    'ego2lidar', 'ego2global', 'gt_boxes', 'gt_labels',
                    'occ_gt_path'])
        ])
]

# Override only the pipeline members of the inherited dataset config.  The
# dataset, sampler, ann_file and all non-actor evaluation behavior remain the
# BaseLine values.
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

evaluation = dict(
    planning_output_path='work_dirs/sparseworld-traj-baseline/eval/output_data.pkl')
