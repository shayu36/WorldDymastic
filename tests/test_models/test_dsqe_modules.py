import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace

from mmdet3d.models.sparsedetectors.bbox.utils import (
    decode_points, encode_points)
from mmdet3d.models.sparsedetectors.dsqe_dual_evolution import (
    DSQEDualEvolution)
from mmdet3d.models.sparsedetectors.dsqe_dual_interaction import (
    DSQEDualInteraction, DSQERoleSpatialAttention)
from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp
from mmdet3d.models.sparsedetectors.dsqe_joint_refine import DSQEJointRefine
from mmdet3d.models.sparsedetectors.dsqe_role_router import DSQERoleRouter
from mmdet3d.models.sparsedetectors.opus_head import OPUSHead
from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import (
    SparseWorld4DTraj)
from mmdet3d.datasets.occ_metrics import Metric_mIoU_Temporal


PC_RANGE = [-10.0, -10.0, -2.0, 10.0, 10.0, 2.0]
DYNAMIC_IDS = [2, 3, 4, 5, 6, 7, 9, 10]


def test_role_router_uses_soft_explicit_class_sets():
    router = DSQERoleRouter(
        embed_dims=16,
        num_classes=17,
        dynamic_class_ids=DYNAMIC_IDS,
        hidden_dims=8)
    query_feat = torch.randn(1, 2, 16, requires_grad=True)
    points = torch.rand(1, 2, 4, 3)
    logits = torch.full((1, 2, 4, 17), -12.0)
    logits[:, 0, :, 4] = 12.0
    logits[:, 1, :, 8] = 12.0  # traffic_cone is static
    source_flag = torch.tensor([[[1.0], [0.0]]])

    output = router(query_feat, points, logits, source_flag)
    assert output['role_pred'][0, 0].mean() > 0.99
    assert output['role_pred'][0, 1].mean() < 0.01
    assert output['query_role'].shape == (1, 2, 1)
    output['query_role'].sum().backward()
    assert query_feat.grad is not None
    assert torch.isfinite(query_feat.grad).all()


def test_ego_warp_translation_rotation_and_inverse():
    warp = DSQEEgoWarp(PC_RANGE, frame_mode='future_ego')
    metric_points = torch.tensor([[[[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]]]])
    points = encode_points(metric_points, torch.tensor(PC_RANGE))

    translation_pose = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    translation = warp.pose_to_matrix(translation_pose)
    translated = decode_points(
        warp.current_to_next(points, translation), torch.tensor(PC_RANGE))
    torch.testing.assert_close(
        translated[..., 0], metric_points[..., 0] - 1.0)

    rotation_pose = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    rotation = warp.pose_to_matrix(rotation_pose)
    rotated = decode_points(
        warp.current_to_next(points, rotation), torch.tensor(PC_RANGE))
    torch.testing.assert_close(
        rotated[0, 0, 1], torch.tensor([0.0, -1.0, 0.0]),
        atol=1e-5, rtol=1e-5)

    identity = warp.compose(translation, warp.inverse(translation))
    torch.testing.assert_close(identity, torch.eye(4).unsqueeze(0))

    current_to_t0 = warp.pose_to_matrix(torch.tensor([
        [5.0, 0.0, 1.0, 0.0]
    ]))
    fixed_t0_delta = torch.tensor([[1.0, 0.0]])
    local_delta = warp.t0_displacement_to_current(
        fixed_t0_delta, current_to_t0)
    torch.testing.assert_close(local_delta, torch.tensor([[0.0, -1.0]]))
    relative = warp.pose_to_matrix(torch.cat([
        local_delta, torch.tensor([[0.0, 1.0]])
    ], dim=-1))
    next_to_t0 = warp.compose(current_to_t0, relative)
    torch.testing.assert_close(
        next_to_t0[:, :2, 3], torch.tensor([[6.0, 0.0]]))

    batched_points = points.repeat(2, 3, 1, 1)
    batched_pose = torch.tensor([
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
    ])
    batched_matrix = warp.pose_to_matrix(batched_pose)
    batched_output = decode_points(
        warp.current_to_next(batched_points, batched_matrix),
        torch.tensor(PC_RANGE))
    expected_x = metric_points[..., 0].expand(2, 3, -1) + \
        batched_points.new_tensor([-1.0, -2.0])[:, None, None]
    torch.testing.assert_close(batched_output[:, :, :, 0], expected_x)


def test_lidar_trajectory_conversion_uses_extrinsic_and_geometry_frame():
    warp = DSQEEgoWarp(PC_RANGE, frame_mode='future_ego')
    identity = warp.identity(1, torch.device('cpu'), torch.float32)
    lidar_to_ego = identity.clone()
    lidar_to_ego[:, 0, 3] = 1.0
    displacement_t0_lidar = torch.tensor([[1.0, 0.0]])
    relative_yaw = torch.tensor([[1.0, 0.0]])

    lidar_delta, ego_pose, ego_relative = \
        warp.trajectory_to_ego_relative(
            displacement_t0_lidar,
            identity,
            lidar_to_ego,
            relative_yaw)
    torch.testing.assert_close(lidar_delta, displacement_t0_lidar)
    torch.testing.assert_close(
        ego_pose[:, :2], torch.tensor([[2.0, -1.0]]),
        atol=1e-6, rtol=1e-6)

    recovered_lidar_delta = torch.matmul(
        lidar_to_ego[:, :3, :3].transpose(-1, -2),
        (ego_relative[:, :3, 3] + torch.matmul(
            ego_relative[:, :3, :3],
            lidar_to_ego[:, :3, 3].unsqueeze(-1)).squeeze(-1) -
         lidar_to_ego[:, :3, 3]).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(
        recovered_lidar_delta[:, :2], displacement_t0_lidar,
        atol=1e-6, rtol=1e-6)

    predicted_cumulative = warp.pose_to_matrix(torch.tensor([
        [0.0, 0.0, 1.0, 0.0]
    ]))
    zero_yaw = torch.tensor([[0.0, 1.0]])
    geometry_delta, _, _ = warp.trajectory_to_ego_relative(
        displacement_t0_lidar, identity, identity, zero_yaw)
    wrong_prediction_delta, _, _ = warp.trajectory_to_ego_relative(
        displacement_t0_lidar, predicted_cumulative, identity, zero_yaw)
    torch.testing.assert_close(
        geometry_delta, torch.tensor([[1.0, 0.0]]),
        atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        wrong_prediction_delta, torch.tensor([[0.0, -1.0]]),
        atol=1e-6, rtol=1e-6)


def test_dual_evolution_preserves_all_points_and_sources():
    num_points = 4
    evolution = DSQEDualEvolution(
        embed_dims=16,
        num_points=num_points,
        pc_range=PC_RANGE,
        beta=0.7)
    warp = DSQEEgoWarp(PC_RANGE, frame_mode='future_ego')
    query_feat = torch.randn(1, 3, 16)
    metric_points = torch.tensor([
        [-2.0, 0.0, 0.0], [-1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0], [1.0, 1.5, 0.0]
    ]).reshape(1, 1, num_points, 3)
    carried = encode_points(metric_points, torch.tensor(PC_RANGE))
    new_metric = metric_points + 2.0
    new_points = encode_points(
        new_metric.expand(1, 2, -1, -1).contiguous(),
        torch.tensor(PC_RANGE))
    identity = warp.identity(1, query_feat.device, query_feat.dtype)
    query_role = torch.tensor([[[0.0], [1.0], [1.0]]])
    point_role = query_role.unsqueeze(2).expand(-1, -1, num_points, -1)

    output = evolution(
        query_feat, carried, new_points, torch.randn(1, 1, 16),
        query_role, point_role, identity, identity, warp)
    assert output['points'].shape == (1, 3, num_points, 3)
    torch.testing.assert_close(
        output['points_metric'][:, :1], metric_points)
    assert torch.unique(output['points_metric'][0, 0], dim=0).shape[0] == num_points

    all_new = evolution(
        query_feat[:, :2], carried[:, :0], new_points,
        torch.randn(1, 1, 16), query_role[:, :2],
        point_role[:, :2], identity, identity, warp)
    assert all_new['points'].shape == (1, 2, num_points, 3)
    assert all_new['query_motion'].shape == (1, 0, 3)


def test_role_attention_empty_sender_is_exactly_zero():
    attention = DSQERoleSpatialAttention(
        embed_dims=16, num_heads=4, local_k=3, dropout=0.0)
    attention.eval()
    query = torch.randn(2, 5, 16)
    empty_stream = torch.zeros_like(query)
    query_gate = torch.ones(2, 5, 1)
    empty_gate = torch.zeros(2, 5, 1)
    centers = torch.randn(2, 5, 3)

    output = attention(
        query, empty_stream, empty_stream,
        query_gate, empty_gate, centers)
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_dual_interaction_and_joint_refine_are_finite():
    interaction = DSQEDualInteraction(
        embed_dims=16, num_heads=4, local_k=3, dropout=0.0)
    joint = DSQEJointRefine(
        embed_dims=16, num_points=4, num_heads=4,
        local_k=3, dropout=0.0)
    interaction.eval()
    joint.eval()

    query_feat = torch.randn(2, 5, 16, requires_grad=True)
    points_metric = torch.randn(2, 5, 4, 3)
    query_role = torch.tensor([
        [[0.0], [0.0], [1.0], [1.0], [0.5]],
        [[1.0], [1.0], [0.0], [0.0], [0.5]],
    ])
    stream = interaction(query_feat, query_role, points_metric)
    output = joint(
        query_feat, stream['dynamic_feat'], stream['static_feat'],
        points_metric)

    assert output['query_feat'].shape == query_feat.shape
    assert output['role_correction'].shape == (2, 5, 4, 1)
    assert torch.isfinite(output['query_feat']).all()
    assert stream['dynamic_from_static_gate'] > \
        stream['static_from_dynamic_gate']
    output['query_feat'].sum().backward()
    assert torch.isfinite(query_feat.grad).all()


def test_semantic_correction_is_an_explicit_scaled_residual():
    model = SimpleNamespace(
        num_refines=4,
        dsqe_cfg={'semantic_residual_scale': 0.25},
        semantic_correction_head=nn.Linear(8, 4 * 17, bias=False))
    nn.init.constant_(model.semantic_correction_head.weight, 0.5)
    base = torch.randn(2, 3, 4, 17)
    feat = torch.ones(2, 3, 8, requires_grad=True)

    semantics, correction = SparseWorld4DTraj._refine_semantics(
        model, base, feat)
    torch.testing.assert_close(semantics, base + 0.25 * correction)
    (semantics - base).sum().backward()
    assert feat.grad is not None
    assert torch.isfinite(feat.grad).all()


def test_chunked_dynamic_nearest_neighbor_matches_full_cdist(monkeypatch):
    torch.manual_seed(7)
    query = torch.randn(23, 3)
    reference = torch.randn(19, 3, requires_grad=True)
    expected = torch.cdist(query, reference.detach(), p=1).argmin(dim=1)

    cdist_shapes = []
    original_cdist = torch.cdist

    def recording_cdist(left, right, *args, **kwargs):
        cdist_shapes.append((left.shape[0], right.shape[0]))
        return original_cdist(left, right, *args, **kwargs)

    monkeypatch.setattr(torch, 'cdist', recording_cdist)
    actual = OPUSHead._chunked_nearest_indices(
        query, reference, chunk_size=5)
    torch.testing.assert_close(actual, expected)
    assert cdist_shapes
    assert max(max(shape) for shape in cdist_shapes) <= 5

    paired = reference[actual]
    (query - paired).abs().mean().backward()
    assert reference.grad is not None
    assert torch.isfinite(reference.grad).all()


def test_official_baseline_foreground_mask_is_preserved():
    pc_range = torch.tensor(PC_RANGE)
    model = SimpleNamespace(pc_range=pc_range)
    model._to_batch_tensor = SparseWorld4DTraj._to_batch_tensor
    model.trans_points = lambda points, delta, matrix: \
        SparseWorld4DTraj.trans_points(model, points, delta, matrix)

    metric_points = torch.tensor([[[
        [-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]
    ]]])
    points = encode_points(metric_points, pc_range)
    trajectory = torch.tensor([[[3.0, 0.0]]])
    kwargs = {'temporal_trajs': trajectory}
    img_metas = [{'ego2lidar': torch.eye(4).numpy()}]

    actual = SparseWorld4DTraj._baseline_foreground_mask(
        model, points, 0, img_metas, kwargs)

    offset = torch.tensor([[[-3.0, 0.0, 0.0]]])
    official_points = SparseWorld4DTraj.trans_points(
        model, points.flatten(1, 2), offset,
        torch.eye(4).unsqueeze(0)).reshape_as(points)
    expected = official_points[..., 0] >= 0
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(
        actual, decode_points(points, pc_range)[..., 0] >= 0)


def test_temporal_metric_reports_dynamic_and_static_group_miou():
    metric = Metric_mIoU_Temporal(num_classes=18)
    perfect_hist = np.eye(18, dtype=np.float64)
    metric.hist_0s = perfect_hist.copy()
    metric.hist_1s = perfect_hist.copy()
    metric.hist_2s = perfect_hist.copy()
    metric.hist_3s = perfect_hist.copy()

    dynamic = metric.count_group_miou(DYNAMIC_IDS)
    static = metric.count_group_miou([1, 8, 11, 12, 13, 14, 15, 16])
    assert dynamic == [100.0] * 4
    assert static == [100.0] * 4
