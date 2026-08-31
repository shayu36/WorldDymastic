#!/usr/bin/env bash
# Start DSQE training with the companion /data/jxy/projects conventions.
#
# Defaults intentionally match the run discussed for this repository:
#   2 GPUs, 2 samples/GPU, gradient accumulation of 4, and at most 32 epochs.
# The model/config remains the current DSQE implementation; the script only
# applies the project-style runtime, optimizer, checkpoint, and BaseLine
# initialization settings at the command line.
#
# Usage (from the repository root):
#   bash tools/train_dsqe_project.sh
#   GPUS=1 CUDA_VISIBLE_DEVICES=0 MAX_EPOCHS=6 bash tools/train_dsqe_project.sh
#   VALIDATE=1 bash tools/train_dsqe_project.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py}"
WORK_DIR="${WORK_DIR:-work_dirs/dsqe-project-32-baseline56-b2}"
GPUS="${GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29526}"
PYTHON_BIN="${PYTHON_BIN:-/data/jxy/projects/env/bin/python3.9}"
BASELINE_CKPT="${BASELINE_CKPT:-/data/jxy/projects/ckpts/epoch_56.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-32}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-2}"
CUMULATIVE_ITERS="${CUMULATIVE_ITERS:-4}"
VALIDATE="${VALIDATE:-0}"

if [[ ! -f "$BASELINE_CKPT" ]]; then
    echo "ERROR: BaseLine checkpoint not found: $BASELINE_CKPT" >&2
    exit 2
fi
if [[ ! "$MAX_EPOCHS" =~ ^[0-9]+$ ]] || (( MAX_EPOCHS < 1 || MAX_EPOCHS > 32 )); then
    echo "ERROR: MAX_EPOCHS must be an integer in [1, 32], got: $MAX_EPOCHS" >&2
    exit 2
fi
if [[ ! "$GPUS" =~ ^[1-9][0-9]*$ ]] || [[ ! "$SAMPLES_PER_GPU" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$CUMULATIVE_ITERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GPUS, SAMPLES_PER_GPU, and CUMULATIVE_ITERS must be positive integers" >&2
    exit 2
fi

cfg_options=(
    runner._delete_=True
    runner.type=EpochBasedRunner
    runner.max_epochs="$MAX_EPOCHS"
    data.samples_per_gpu="$SAMPLES_PER_GPU"
    data.workers_per_gpu=4
    model.img_backbone.with_cp=True
    optimizer.type=AdamW
    optimizer.lr=1e-4
    optimizer.weight_decay=1e-2
    optimizer_config.type=GradientCumulativeOptimizerHook
    optimizer_config.cumulative_iters="$CUMULATIVE_ITERS"
    optimizer_config.grad_clip.max_norm=5
    optimizer_config.grad_clip.norm_type=2
    checkpoint_config.interval=1
    checkpoint_config.max_keep_ckpts=-1
    checkpoint_config.save_last=True
    evaluation.interval=1
    evaluation.planning_output_path="$WORK_DIR/eval/output_data.pkl"
    log_config.interval=50
    find_unused_parameters=False
    load_from="$BASELINE_CKPT"
)

extra_args=()
if [[ "$VALIDATE" == "1" ]]; then
    extra_args+=(--validate)
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
    extra_args+=(--resume-from "$RESUME_FROM")
fi

echo "Configuration: $CONFIG"
echo "Work directory: $WORK_DIR"
echo "BaseLine checkpoint: $BASELINE_CKPT"
echo "GPUs: $GPUS; samples/GPU: $SAMPLES_PER_GPU; cumulative_iters: $CUMULATIVE_ITERS"
echo "Maximum epochs: $MAX_EPOCHS"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node="$GPUS" \
    --master_port="$MASTER_PORT" \
    tools/train.py \
    "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher pytorch \
    "${extra_args[@]}" \
    --cfg-options "${cfg_options[@]}" \
    "$@"
