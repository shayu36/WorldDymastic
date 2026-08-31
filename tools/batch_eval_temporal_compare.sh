#!/usr/bin/env bash
# Evaluate all retained temporal checkpoints and compare them with BaseLine.
#
# This follows /data/jxy/projects/tools/batch_eval_all_epochs.sh, but calls
# this repository's tools/test.py so DSQE uses the dictionary-aware temporal
# collector and the memory-safe uint8 distributed implementation.  Single GPU
# is the default because full nuScenes validation is host-memory intensive.
#
# Usage (from the repository root):
#   bash tools/batch_eval_temporal_compare.sh
#   GPU_ID=1 WORK_DIR=work_dirs/my_run bash tools/batch_eval_temporal_compare.sh
#   EPOCHS="6 9 12 15 18 21 32" bash tools/batch_eval_temporal_compare.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py}"
WORK_DIR="${WORK_DIR:-work_dirs/dsqe-ddp-32-baseline56-b2}"
RESULT_DIR="${RESULT_DIR:-${WORK_DIR}/eval_results}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/data/jxy/projects/env/bin/python3.9}"
BASELINE_LOG="${BASELINE_LOG:-/data/jxy/projects/work_dirs/sparseworld-traj-memory-only/eval_epoch56_memory_off.log}"
SELECT_EPOCHS="${EPOCHS:-}"

mkdir -p "$RESULT_DIR"

if [[ ! -f "$BASELINE_LOG" ]]; then
    echo "ERROR: BaseLine log not found: $BASELINE_LOG" >&2
    exit 2
fi

shopt -s nullglob
checkpoints=("$WORK_DIR"/epoch_*.pth)
if (( ${#checkpoints[@]} == 0 )); then
    echo "ERROR: no epoch_*.pth checkpoints found in $WORK_DIR" >&2
    exit 2
fi

# Keep concurrent runs for the same work directory from writing the same
# evaluation logs/results.  The lock is removed automatically on exit.
lock_name="${WORK_DIR//\//_}"
lock_file="/tmp/batch_eval_temporal_${lock_name}.lock"
if [[ -f "$lock_file" ]]; then
    lock_pid="$(cat "$lock_file" 2>/dev/null || true)"
    if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
        echo "ERROR: another evaluation is already running (PID $lock_pid)" >&2
        exit 1
    fi
    rm -f "$lock_file"
fi
echo "$$" > "$lock_file"
trap 'rm -f "$lock_file"' EXIT

eval_logs=()
for checkpoint in $(printf '%s\n' "${checkpoints[@]}" | sort -V); do
    checkpoint_name="$(basename "$checkpoint" .pth)"
    epoch="${checkpoint_name#epoch_}"
    if [[ -n "$SELECT_EPOCHS" ]] && \
       [[ " $SELECT_EPOCHS " != *" $epoch "* ]]; then
        continue
    fi
    log_file="$RESULT_DIR/epoch_${epoch}_eval.log"
    planning_output="$RESULT_DIR/output_data_epoch_${epoch}.pkl"

    # An evaluation is complete only when the final metric dictionary was
    # printed.  Progress output alone must not cause a checkpoint to be
    # skipped after an interrupted test.
    if [[ -f "$log_file" ]] && \
       grep -q "{'IoU':" "$log_file" && \
       grep -q "'mIoU':" "$log_file"; then
        echo "Epoch $epoch already has a final metric dictionary; skipping."
        eval_logs+=("$log_file")
        continue
    fi

    echo "============================================"
    echo "Evaluating $checkpoint_name: $checkpoint (GPU $GPU_ID)"
    echo "============================================"

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" tools/test.py \
        --config "$CONFIG" \
        --checkpoint "$checkpoint" \
        --gpu-id 0 \
        --eval segm \
        --deterministic \
        --eval-options "planning_output_path=$planning_output" \
        2>&1 | tee "$log_file"

    eval_logs+=("$log_file")
    "$PYTHON_BIN" -c 'import torch; torch.cuda.empty_cache()' 2>/dev/null || true
done

if (( ${#eval_logs[@]} == 0 )); then
    echo "ERROR: no checkpoints selected for evaluation (EPOCHS=$SELECT_EPOCHS)" >&2
    exit 2
fi

compare_args=(
    --baseline-log "$BASELINE_LOG"
    --output-dir "$RESULT_DIR"
    --work-dir "$WORK_DIR"
    --model DSQE
)
for log_file in "${eval_logs[@]}"; do
    compare_args+=(--log "$log_file")
done

echo "============================================"
echo "Generating DSQE versus BaseLine summary"
echo "============================================"
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" tools/compare_temporal_eval.py "${compare_args[@]}"
