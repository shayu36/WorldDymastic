#!/usr/bin/env bash
# Start DSQE training with the companion /data/jxy/projects conventions.
#
# Defaults intentionally preserve the historical run:
#   2 GPUs, 2 samples/GPU, gradient accumulation of 4, and at most 32 epochs.
# Pilot runs can change throughput without editing the model config, e.g.
#   SAMPLES_PER_GPU=4 CUMULATIVE_ITERS=2 LOAD_INTERVAL=20 \
#   UNFREEZE_TASS=0 MAX_EPOCHS=5 bash tools/train_dsqe_project.sh
# The default model is DSQE-PreSCF; pass CONFIG=... to run the frozen BaseLine
# comparison.  The script applies project-style runtime, optimizer,
# checkpoint, and BaseLine initialization settings at the command line.
#
# Usage (from the repository root):
#   bash tools/train_dsqe_project.sh
#   GPUS=1 CUDA_VISIBLE_DEVICES=0 MAX_EPOCHS=6 bash tools/train_dsqe_project.sh
#   VALIDATE=1 bash tools/train_dsqe_project.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/sparseworld/nuscenes-temporal/sparseworld-traj-prescf.py}"
WORK_DIR="${WORK_DIR:-work_dirs/dsqe-prescf-32-baseline56-b2}"
GPUS="${GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29526}"
PYTHON_BIN="${PYTHON_BIN:-/data/jxy/projects/env/bin/python3.9}"
BASELINE_CKPT="${BASELINE_CKPT:-/data/jxy/projects/ckpts/epoch_56.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-32}"
RUNNER_TYPE="${RUNNER_TYPE:-EpochBasedRunner}"
MAX_ITERS="${MAX_ITERS:-}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-2}"
CUMULATIVE_ITERS="${CUMULATIVE_ITERS:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
LOAD_INTERVAL="${LOAD_INTERVAL:-1}"
FORECAST_STEPS="${FORECAST_STEPS:-}"
UNFREEZE_TASS="${UNFREEZE_TASS:-}"
SKIP_FROZEN_BASELINE_LOSSES="${SKIP_FROZEN_BASELINE_LOSSES:-}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"
VALIDATE="${VALIDATE:-0}"

# Accept the older positional override style as well.  Without this
# normalization, ``runner.type=IterBasedRunner runner.max_iters=50`` would be
# merged after the script's default ``runner.max_epochs`` and MMCV would
# reject the configuration because both limits are present.
for override in "$@"; do
    case "$override" in
        runner.type=IterBasedRunner)
            RUNNER_TYPE="IterBasedRunner" ;;
        runner.type=EpochBasedRunner)
            RUNNER_TYPE="EpochBasedRunner" ;;
        runner.max_iters=*)
            MAX_ITERS="${override#runner.max_iters=}" ;;
        runner.max_epochs=*)
            MAX_EPOCHS="${override#runner.max_epochs=}" ;;
    esac
done

if [[ ! -f "$BASELINE_CKPT" ]]; then
    echo "ERROR: BaseLine checkpoint not found: $BASELINE_CKPT" >&2
    exit 2
fi
if [[ "$RUNNER_TYPE" != "EpochBasedRunner" && "$RUNNER_TYPE" != "IterBasedRunner" ]]; then
    echo "ERROR: RUNNER_TYPE must be EpochBasedRunner or IterBasedRunner" >&2
    exit 2
fi
if [[ "$RUNNER_TYPE" == "EpochBasedRunner" ]] && \
   { [[ ! "$MAX_EPOCHS" =~ ^[0-9]+$ ]] || (( MAX_EPOCHS < 1 || MAX_EPOCHS > 32 )); }; then
    echo "ERROR: MAX_EPOCHS must be an integer in [1, 32], got: $MAX_EPOCHS" >&2
    exit 2
fi
if [[ "$RUNNER_TYPE" == "IterBasedRunner" ]] && \
   { [[ ! "$MAX_ITERS" =~ ^[1-9][0-9]*$ ]]; }; then
    echo "ERROR: MAX_ITERS must be a positive integer for IterBasedRunner" >&2
    exit 2
fi
if [[ ! "$GPUS" =~ ^[1-9][0-9]*$ ]] || [[ ! "$SAMPLES_PER_GPU" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$CUMULATIVE_ITERS" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$LOAD_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GPUS, SAMPLES_PER_GPU, CUMULATIVE_ITERS, WORKERS_PER_GPU, and LOAD_INTERVAL must be positive integers" >&2
    exit 2
fi
if [[ "$CUDNN_BENCHMARK" != "0" && "$CUDNN_BENCHMARK" != "1" ]]; then
    echo "ERROR: CUDNN_BENCHMARK must be 0 or 1" >&2
    exit 2
fi

cfg_options=(
    runner._delete_=True
    runner.type="$RUNNER_TYPE"
    data.samples_per_gpu="$SAMPLES_PER_GPU"
    data.workers_per_gpu="$WORKERS_PER_GPU"
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
if [[ "$RUNNER_TYPE" == "EpochBasedRunner" ]]; then
    cfg_options+=(runner.max_epochs="$MAX_EPOCHS")
else
    cfg_options+=(runner.max_iters="$MAX_ITERS")
fi

if [[ "$CUDNN_BENCHMARK" == "1" ]]; then
    cfg_options+=(cudnn_benchmark=True)
fi
if [[ -n "$FORECAST_STEPS" ]]; then
    cfg_options+=(model.dsqe_cfg.forecast_steps="$FORECAST_STEPS")
fi
if [[ -n "$UNFREEZE_TASS" ]]; then
    cfg_options+=(model.dsqe_cfg.unfreeze_tass="$UNFREEZE_TASS")
fi
if [[ -n "$SKIP_FROZEN_BASELINE_LOSSES" ]]; then
    cfg_options+=(model.dsqe_cfg.skip_frozen_baseline_losses="$SKIP_FROZEN_BASELINE_LOSSES")
fi

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
echo "GPUs: $GPUS; samples/GPU: $SAMPLES_PER_GPU; cumulative_iters: $CUMULATIVE_ITERS; workers/GPU: $WORKERS_PER_GPU"
echo "Load interval: $LOAD_INTERVAL; forecast steps: ${FORECAST_STEPS:-config}; unfreeze TASS: ${UNFREEZE_TASS:-config}"
if [[ "$RUNNER_TYPE" == "EpochBasedRunner" ]]; then
    echo "Maximum epochs: $MAX_EPOCHS"
else
    echo "Maximum iterations: $MAX_ITERS"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node="$GPUS" \
    --master_port="$MASTER_PORT" \
    tools/train.py \
    "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher pytorch \
    --load_interval "$LOAD_INTERVAL" \
    "${extra_args[@]}" \
    --cfg-options "${cfg_options[@]}" \
    "$@"
