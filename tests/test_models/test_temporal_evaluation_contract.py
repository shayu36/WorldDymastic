import pickle

import numpy as np
import pytest
import torch

import mmdet3d.apis.test as test_api
from mmdet3d.apis.test import (_format_temporal_prediction,
                               _split_temporal_collected_results)
from mmdet3d.datasets.nuscenes_dataset_occ_trajectory import (
    _dump_planning_results, _unpack_temporal_evaluation_results)


def _model_result(value):
    result = {
        f'semantic_occ_{horizon}s': [
            np.full((2, 3, 1), value + horizon, dtype=np.uint8)
        ]
        for horizon in (0, 2, 4, 6)
    }
    result['pred_traj'] = torch.full((1, 6, 2), float(value))
    return result


def test_temporal_prediction_contract_preserves_occupancy_and_trajectory():
    occupancy, trajectory = _format_temporal_prediction(_model_result(3))

    assert occupancy.shape == (4, 2, 3, 1)
    assert occupancy.dtype == np.uint8
    assert trajectory.shape == (1, 6, 2)
    assert trajectory.device.type == 'cpu'

    paired = _split_temporal_collected_results([
        (occupancy, trajectory), (occupancy + 1, trajectory + 1)
    ])
    unpacked_occupancy, unpacked_trajectory = \
        _unpack_temporal_evaluation_results(paired)

    assert len(unpacked_occupancy) == 2
    assert len(unpacked_trajectory) == 2
    np.testing.assert_array_equal(unpacked_occupancy[1], occupancy + 1)
    torch.testing.assert_close(unpacked_trajectory[1], trajectory + 1)


def test_temporal_result_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match='result counts must match'):
        _unpack_temporal_evaluation_results([[np.zeros(1), np.zeros(1)],
                                             [torch.zeros(1)]])


def test_planning_output_is_written_to_configured_path(tmp_path):
    output_path = tmp_path / 'nested' / 'output_data.pkl'
    expected = {'sample-token': np.ones((1, 6, 2), dtype=np.float32)}

    actual_path = _dump_planning_results(expected, str(output_path))

    assert actual_path == str(output_path.resolve())
    with output_path.open('rb') as file:
        actual = pickle.load(file)
    np.testing.assert_array_equal(actual['sample-token'],
                                  expected['sample-token'])


def test_cpu_result_collection_honors_explicit_tmpdir(tmp_path, monkeypatch):
    collect_dir = tmp_path / 'distributed-results'
    monkeypatch.setattr(test_api, 'get_dist_info', lambda: (0, 1))
    monkeypatch.setattr(test_api.dist, 'barrier', lambda: None)

    results = test_api.collect_results_cpu(
        [{'sample': 1}], size=1, tmpdir=str(collect_dir))

    assert results == [{'sample': 1}]
    assert collect_dir.is_dir()
    assert list(collect_dir.iterdir()) == []
