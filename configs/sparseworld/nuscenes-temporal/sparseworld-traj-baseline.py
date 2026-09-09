"""Frozen reference configuration for the original SparseWorld SCF path."""

_base_ = ['./sparseworld-traj-finetune.py']

model = dict(
    dsqe_mode='baseline',
    dsqe_cfg=dict(enabled=False, mode='baseline'))

evaluation = dict(
    planning_output_path='work_dirs/sparseworld-traj-baseline/eval/output_data.pkl')
