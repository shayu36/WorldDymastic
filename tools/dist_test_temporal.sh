#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=$((RANDOM + 10000))
PYTHON_BIN=${PYTHON_BIN:-python}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
"$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test_temporal.py \
    $CONFIG \
    $CHECKPOINT \
    --eval segm \
    --launcher pytorch \
    ${@:4}
