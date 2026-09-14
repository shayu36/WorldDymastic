"""DSQE-PreSCF training configuration.

The BaseLine configuration remains untouched and can be used for checkpoint
and metric comparisons.  PreSCF replaces the future SCF recursion with a
single carried/new Query state updated by the role, warp, evolution,
interaction and joint-refinement modules.
"""

_base_ = ['./sparseworld-traj-baseline.py']

model = dict(
    dsqe_mode='prescf',
    dsqe_cfg=dict(
        enabled=True,
        mode='prescf',
        dynamic_class_ids=[2, 3, 4, 5, 6, 7, 9, 10],
        static_class_ids=[1, 8, 11, 12, 13, 14, 15, 16],
        forecast_steps=None,
        # Stage 1 starts with GT-routed roles/ego transforms and linearly
        # decays both signals to fully predicted closed-loop rollout.
        teacher_forcing=1.0,
        role_teacher_forcing=1.0,
        ego_teacher_forcing=1.0,
        teacher_forcing_start_epoch=5,
        teacher_forcing_end_epoch=16,
        stage3_start_epoch=16,
        unfreeze_tass=True,
        unfreeze_tass_layers=2,
        role_box_inflation=0.5,
        # The temporal actor cache stores native nuScenes ``(w,l,h)`` sizes.
        role_box_dims_order='wlh',
        # nuScenes future actor trajectories are absolute displacements from
        # the current ego frame at each horizon (not per-step increments).
        role_trajectory_mode='absolute',
        role_dynamic_weight='auto',
        role_dynamic_weight_max=20.0,
        role_focal_gamma=2.0,
        lambda_role=0.1,
        lambda_ego=0.1,
        lambda_static=0.01,
        lambda_dynamic=0.01,
        lambda_smooth=0.01,
        freeze_backbone=True,
        freeze_baseline_heads=True,
        freeze_tass=True))

evaluation = dict(
    planning_output_path='work_dirs/sparseworld-traj-prescf/eval/output_data.pkl')

# TASS is frozen in Stages 1/2.  Its final decoder layers enter the optimizer
# at a lower LR and become trainable only when ``stage3_start_epoch`` is met.
optimizer = dict(
    paramwise_cfg=dict(custom_keys={'pts_bbox_head': dict(lr_mult=0.1)}))
