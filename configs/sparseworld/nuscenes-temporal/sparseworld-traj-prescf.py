"""DSQE-PreSCF training configuration.

The BaseLine execution path remains independent and can be used for
checkpoint/metric comparisons.  PreSCF replaces the future SCF recursion with
a single carried/new Query state updated by the role, warp, evolution,
interaction and joint-refinement modules.
"""

# PreSCF needs the actor-only supervision fields that the pure BaseLine
# reference deliberately omits.  Both configurations still share the same
# model/data definitions from the finetune base, while the execution mode is
# selected explicitly below.
_base_ = ['./sparseworld-traj-finetune.py']

model = dict(
    dsqe_mode='prescf',
    dsqe_cfg=dict(
        enabled=True,
        mode='prescf',
        dynamic_class_ids=[2, 3, 4, 5, 6, 7, 9, 10],
        static_class_ids=[1, 8, 11, 12, 13, 14, 15, 16],
        forecast_steps=None,
        forecast_curriculum=[1, 2, 3, 6],
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
        # VAD concatenates nuScenes ``Box.wlh`` into ``gt_boxes``.  SECOND
        # yaw's local x/y axes pair with that native width/length ordering.
        role_box_dims_order='wlh',
        role_yaw_encoding='raw',
        # VAD/nuScenes caches store adjacent actor displacements in the
        # current LiDAR frame.  PreSCF cumulatively integrates them after the
        # explicit LiDAR->E0 ego conversion.
        role_trajectory_mode='increment',
        role_dynamic_weight='auto',
        role_dynamic_weight_max=20.0,
        role_focal_gamma=2.0,
        role_match_max_distance=2.5,
        # Unassociated points may acquire (but never replace) a persistent
        # dynamic actor ID once future geometry enters this metric radius.
        actor_recovery_max_distance=2.5,
        # Bidirectional dynamic coverage uses a robust pointwise Huber loss;
        # this beta is in metric metres after Query points are decoded.
        dynamic_huber_beta=0.2,
        planar_motion_only=False,
        dynamic_semantic_weight=0.25,
        lambda_role=0.1,
        lambda_ego=0.1,
        lambda_static=0.01,
        lambda_dynamic=0.01,
        lambda_smooth=0.01,
        lambda_leak=0.01,
        stage1_end_epoch=5,
        stage2_end_epoch=16,
        interaction_ramp_epochs=4,
        joint_ramp_epochs=4,
        stage_gate_floor=0.1,
        joint_correction_min_gate=0.1,
        freeze_backbone=True,
        freeze_baseline_heads=True,
        freeze_tass=True))

evaluation = dict(
    planning_output_path='work_dirs/sparseworld-traj-prescf/eval/output_data.pkl')

# TASS is frozen in Stages 1/2.  Its final decoder layers enter the optimizer
# at a lower LR and become trainable only when ``stage3_start_epoch`` is met.
optimizer = dict(
    paramwise_cfg=dict(custom_keys={
        # The Stage-3 decoder layers are DDP-visible from initialization but
        # receive zero gradients during Stages 1/2.  Zero decay prevents
        # AdamW from changing them while that gradient gate is closed.
        'pts_bbox_head': dict(lr_mult=0.1, decay_mult=0.0),
    }))
