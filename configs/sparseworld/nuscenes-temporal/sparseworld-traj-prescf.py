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
        teacher_forcing=0.0,
        lambda_role=0.1,
        lambda_ego=0.1,
        lambda_static=0.01,
        lambda_dynamic=0.01,
        lambda_smooth=0.01,
        freeze_backbone=True,
        freeze_tass=True))

evaluation = dict(
    planning_output_path='work_dirs/sparseworld-traj-prescf/eval/output_data.pkl')
