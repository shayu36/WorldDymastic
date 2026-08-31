"""Evaluation hooks for SparseWorld temporal dictionary outputs.

The generic MMDetection hooks expect the test API to return the standard
detector result structure.  ``SparseWorld4DTraj`` instead returns temporal
occupancy and trajectory dictionaries, so validation must use the temporal
collector from :mod:`mmdet3d.apis.test`.  This module mirrors the dedicated
hooks used by the companion ``/data/jxy/projects`` training stack while
keeping this repository's memory-safe ``uint8`` collector.
"""

import os.path as osp
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmdet.core import DistEvalHook, EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


_TEMPORAL_HORIZONS = ("0s", "1s", "2s", "3s")


def _as_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float)):
        return value
    return None


def _flatten_eval_results(eval_results):
    """Convert temporal metric lists into scalar logger fields."""

    flattened = {}
    for name, value in eval_results.items():
        scalar = _as_scalar(value)
        if scalar is not None:
            flattened[name] = scalar
            continue

        if isinstance(value, (list, tuple, np.ndarray)):
            values = list(value)
            labels = _TEMPORAL_HORIZONS if len(values) == 4 else tuple(
                str(index) for index in range(len(values)))
            numeric_values = []
            for label, item in zip(labels, values):
                item_scalar = _as_scalar(item)
                if item_scalar is None:
                    raise TypeError(
                        f"Evaluation metric {name!r} contains a non-scalar "
                        f"value of type {type(item).__name__}")
                flattened[f"{name}_{label}"] = item_scalar
                numeric_values.append(float(item_scalar))
            if numeric_values:
                flattened[f"{name}_mean"] = float(np.mean(numeric_values))
                if len(numeric_values) == 4:
                    flattened[f"{name}_future_mean"] = float(
                        np.mean(numeric_values[1:]))
            continue

        raise TypeError(
            f"Evaluation metric {name!r} has unsupported type "
            f"{type(value).__name__}")
    return flattened


class _SparseWorldMetricLoggingMixin:

    def evaluate(self, runner, results):
        raw_eval_results = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        eval_results = _flatten_eval_results(raw_eval_results)

        runner.log_buffer.output.update(eval_results)
        runner.log_buffer.ready = True

        if self.save_best is not None:
            if not eval_results:
                warnings.warn(
                    "Since eval_results is empty, save_best will be skipped.")
                return None
            if self.key_indicator == "auto":
                self._init_rule(self.rule, list(eval_results.keys())[0])
            if self.key_indicator not in eval_results:
                raise KeyError(
                    f"save_best metric {self.key_indicator!r} is unavailable; "
                    f"available scalar metrics: {list(eval_results)}")
            return eval_results[self.key_indicator]
        return None


class SparseWorldEvalHook(_SparseWorldMetricLoggingMixin, EvalHook):
    """Single-GPU validation hook for SparseWorld dictionary outputs."""

    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return

        # Import lazily to avoid an API/evaluation import cycle at startup.
        from mmdet3d.apis.test import single_gpu_test

        results = single_gpu_test(runner.model, self.dataloader)
        self.latest_results = results
        runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)


class SparseWorldDistEvalHook(_SparseWorldMetricLoggingMixin, DistEvalHook):
    """Distributed validation hook for SparseWorld dictionary outputs."""

    def _do_evaluate(self, runner):
        if self.broadcast_bn_buffer:
            for module in runner.model.modules():
                if isinstance(module, _BatchNorm) and \
                        module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, ".eval_hook")

        # Keep the repository's temporal collector: occupancy is converted to
        # uint8 before CPU gathering so validation does not exhaust host RAM.
        from mmdet3d.apis.test import multi_gpu_test_temporal

        results = multi_gpu_test_temporal(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        self.latest_results = results
        if runner.rank == 0:
            print("\n")
            runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)
