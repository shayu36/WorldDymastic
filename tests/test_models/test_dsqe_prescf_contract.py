"""Small, dependency-light acceptance tests for the PreSCF state contract."""

from pathlib import Path

import pytest


def test_baseline_and_prescf_configs_are_separate():
    root = Path(__file__).parents[2]
    baseline = root / 'configs/sparseworld/nuscenes-temporal/sparseworld-traj-baseline.py'
    prescf = root / 'configs/sparseworld/nuscenes-temporal/sparseworld-traj-prescf.py'
    assert baseline.exists() and prescf.exists()
    assert "dsqe_mode='baseline'" in baseline.read_text()
    assert "dsqe_mode='prescf'" in prescf.read_text()


def test_prescf_source_is_recursive_and_has_no_residual_anchor():
    root = Path(__file__).parents[2]
    source = (root / 'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py').read_text()
    assert 'def _forward_prescf' in source
    assert 'def _forward_baseline_scf' in source
    assert 'base_points + residual' not in source
    assert '_baseline_future_step' not in source


@pytest.mark.parametrize('num_carried,num_new', [(2, 0), (0, 2), (1, 2)])
def test_evolution_preserves_point_shapes_and_motion(num_carried, num_new):
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.bbox.utils import encode_points
    from mmdet3d.models.sparsedetectors.dsqe_dual_evolution import DSQEDualEvolution
    from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp

    pc_range = torch.tensor([-10., -10., -2., 10., 10., 2.])
    module = DSQEDualEvolution(8, 4, pc_range, motion_scale=2.)
    warp = DSQEEgoWarp(pc_range)
    feat = torch.zeros(1, num_carried + num_new, 8)
    carried = encode_points(torch.zeros(1, num_carried, 4, 3), pc_range)
    new = encode_points(torch.ones(1, num_new, 4, 3), pc_range)
    role = torch.ones(1, num_carried + num_new, 4, 1) * .5
    query_role = torch.ones(1, num_carried + num_new, 1) * .5
    identity = warp.identity(1, feat.device, feat.dtype)
    result = module(feat, carried, new, torch.zeros(1, 1, 8), query_role,
                    role, identity, identity, warp)
    assert result['points'].shape == (1, num_carried + num_new, 4, 3)
    assert result['query_motion'].shape == (1, num_carried, 3)


def test_query_motion_changes_final_geometry():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.bbox.utils import decode_points, encode_points
    from mmdet3d.models.sparsedetectors.dsqe_dual_evolution import DSQEDualEvolution
    from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp

    pc_range = torch.tensor([-10., -10., -2., 10., 10., 2.])
    module = DSQEDualEvolution(4, 2, pc_range, motion_scale=2.)
    with torch.no_grad():
        module.motion_head[-1].bias.copy_(torch.tensor([1., 0., 0.]))
    warp = DSQEEgoWarp(pc_range)
    points = encode_points(torch.zeros(1, 1, 2, 3), pc_range)
    identity = warp.identity(1, points.device, points.dtype)
    role = torch.ones(1, 1, 2, 1)
    out = module(torch.zeros(1, 1, 4), points, points[:, :0],
                 torch.zeros(1, 1, 4), torch.ones(1, 1, 1), role,
                 identity, identity, warp)
    assert out['query_motion'].abs().sum() > 0
    assert decode_points(out['points'], pc_range).abs().sum() > 0
