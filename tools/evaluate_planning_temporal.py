#!/usr/bin/env python3
"""Evaluate temporal trajectory pickles with the project ST-P3 metric.

The project convention reports one value at 1s, 2s, and 3s.  Each value is
the mean over all trajectory samples and all 2/4/6 points up to that horizon;
there is intentionally no additional ``Avg`` column.  Collision values are
reported in percent and use ``plan_obj_box_col`` from the ST-P3 metric.

Example::

    python tools/evaluate_planning_temporal.py \
        --prediction work_dirs/.../eval_results/output_data_epoch_32.pkl \
        --label epoch_32
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from skimage.draw import polygon


DEFAULT_ROOT = Path("/data/jxy/projects/admlp/stp3_val")
HORIZONS = (1, 2, 3)
POINTS_PER_SECOND = 2


def _load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _as_prediction(value) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 or tensor.shape[1] < 2:
        raise ValueError(
            f"Prediction has unsupported shape {tuple(tensor.shape)}; "
            "expected (6, 2) or (1, 6, 2)."
        )
    return tensor[:, :2]


def _as_occupancy(value) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(
            f"Occupancy has unsupported shape {tuple(tensor.shape)}; "
            "expected (6, 200, 200) or (1, 6, 200, 200)."
        )
    return tensor.bool()


def _gt_trajectory(value) -> torch.Tensor:
    if not isinstance(value, Mapping) or "gt_trajectory" not in value:
        raise ValueError("Ground-truth entry has no gt_trajectory field")
    tensor = torch.as_tensor(
        np.asarray(value["gt_trajectory"]), dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] < 2 or tensor.shape[0] < 7:
        raise ValueError(
            f"Ground-truth trajectory has unsupported shape {tuple(tensor.shape)}"
        )
    # ST-P3 stores the current point first; planning starts at point 1.
    return tensor[1:7, :2]


def _ego_box_collision(
    trajectory: torch.Tensor,
    occupancy: torch.Tensor,
    dx: np.ndarray,
    bx: np.ndarray,
    bev_dimension: np.ndarray,
) -> torch.Tensor:
    """Return per-timestep collision of the ego box with occupancy."""

    ego_width, ego_length = 1.85, 4.084
    points = np.array(
        [
            [-ego_length / 2.0 + 0.5, ego_width / 2.0],
            [ego_length / 2.0 + 0.5, ego_width / 2.0],
            [ego_length / 2.0 + 0.5, -ego_width / 2.0],
            [-ego_length / 2.0 + 0.5, -ego_width / 2.0],
        ]
    )
    points = (points - bx) / dx
    points[:, [0, 1]] = points[:, [1, 0]]
    rows, cols = polygon(points[:, 1], points[:, 0])
    ego_pixels = np.concatenate([rows[:, None], cols[:, None]], axis=-1)

    n_future = trajectory.shape[0]
    points = trajectory.reshape(n_future, 1, 2).clone()
    points[:, :, [0, 1]] = points[:, :, [1, 0]]
    points = points.cpu().numpy() / dx + ego_pixels
    rows = np.clip(points[:, :, 0].astype(np.int32), 0, bev_dimension[0] - 1)
    cols = np.clip(points[:, :, 1].astype(np.int32), 0, bev_dimension[1] - 1)

    collisions = np.zeros(n_future, dtype=bool)
    for timestep in range(n_future):
        collisions[timestep] = np.any(
            occupancy[timestep, rows[timestep], cols[timestep]].cpu().numpy()
        )
    return torch.from_numpy(collisions)


def _collision_counts(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    occupancy: torch.Tensor,
) -> torch.Tensor:
    """Return per-timestep ``plan_obj_box_col`` counts for one sample."""

    dx = np.array([0.5, 0.5], dtype=np.float32)
    bx = np.array([-49.75, -49.75], dtype=np.float32)
    bev_dimension = np.array([200, 200], dtype=np.int32)

    # This sign/sampling convention is the one used by the project's
    # AD-MLP/ST-P3 PlanningMetric implementation.
    prediction = prediction * torch.tensor([-1.0, 1.0])
    ground_truth = ground_truth * torch.tensor([-1.0, 1.0])
    gt_box_collision = _ego_box_collision(
        ground_truth, occupancy, dx, bx, bev_dimension
    )
    predicted_box_collision = _ego_box_collision(
        prediction, occupancy, dx, bx, bev_dimension
    )
    return predicted_box_collision & ~gt_box_collision


def evaluate_prediction(
    prediction_path: Path,
    occupancy_path: Path,
    ground_truth_path: Path,
) -> Dict[str, float]:
    predictions = _load(prediction_path)
    occupancies = _load(occupancy_path)
    ground_truths = _load(ground_truth_path)

    if not isinstance(predictions, Mapping):
        raise ValueError(f"{prediction_path}: expected token-to-trajectory dict")
    keys = [key for key in predictions if key in occupancies and key in ground_truths]
    if not keys:
        raise ValueError("No common prediction/occupancy/ground-truth tokens")

    missing = len(predictions) - len(keys)
    if missing:
        print(f"warning: skipped {missing} prediction tokens without ST-P3 labels")

    l2_sum = torch.zeros(6, dtype=torch.float64)
    collision_sum = torch.zeros(6, dtype=torch.float64)
    for key in keys:
        prediction = _as_prediction(predictions[key])
        ground_truth = _gt_trajectory(ground_truths[key])
        occupancy = _as_occupancy(occupancies[key])
        if prediction.shape[0] < 6 or occupancy.shape[0] < 6:
            raise ValueError(f"{key}: expected six future points")
        prediction = prediction[:6]
        occupancy = occupancy[:6]
        l2_sum += torch.linalg.vector_norm(prediction - ground_truth, dim=-1).double()
        collision_sum += _collision_counts(
            prediction, ground_truth, occupancy
        ).double()

    sample_count = float(len(keys))
    result: Dict[str, float] = {"samples": int(sample_count)}
    for horizon in HORIZONS:
        count = POINTS_PER_SECOND * horizon
        result[f"L2_{horizon}s"] = float(l2_sum[:count].sum() / sample_count / count)
        result[f"plan_obj_box_col_{horizon}s"] = float(
            collision_sum[:count].sum() / sample_count / count * 100.0
        )
    return result


def _write_csv(path: Path, label: str, values: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["checkpoint", "metric", "1s", "2s", "3s"])
        for metric in ("L2 (m)", "plan_obj_box_col (%)"):
            prefix = "L2" if metric.startswith("L2") else "plan_obj_box_col"
            writer.writerow(
                [
                    label,
                    metric,
                    *[f"{values[f'{prefix}_{horizon}s']:.4f}" for horizon in HORIZONS],
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--occupancy", type=Path, default=DEFAULT_ROOT / "stp3_occupancy.pkl")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_ROOT / "stp3_traj_gt.pkl")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    label = args.label or args.prediction.stem
    values = evaluate_prediction(args.prediction, args.occupancy, args.ground_truth)

    print("指标 | 1s | 2s | 3s")
    print("mIoU (%) | n/a | n/a | n/a")
    print(
        "IoU (%) | n/a | n/a | n/a\n"
        f"L2 (m) | {values['L2_1s']:.4f} | {values['L2_2s']:.4f} | {values['L2_3s']:.4f}\n"
        "碰撞率 plan_obj_box_col (%) | "
        f"{values['plan_obj_box_col_1s']:.4f} | "
        f"{values['plan_obj_box_col_2s']:.4f} | "
        f"{values['plan_obj_box_col_3s']:.4f}"
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"checkpoint": label, **values}, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.output_csv:
        _write_csv(args.output_csv, label, values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
