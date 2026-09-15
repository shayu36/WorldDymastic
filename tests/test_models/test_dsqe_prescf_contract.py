"""Small, dependency-light acceptance tests for the PreSCF state contract."""

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_baseline_and_prescf_configs_are_separate():
    root = Path(__file__).parents[2]
    baseline = root / 'configs/sparseworld/nuscenes-temporal/sparseworld-traj-baseline.py'
    prescf = root / 'configs/sparseworld/nuscenes-temporal/sparseworld-traj-prescf.py'
    assert baseline.exists() and prescf.exists()
    assert "dsqe_mode='baseline'" in baseline.read_text()
    assert "dsqe_mode='prescf'" in prescf.read_text()
    from mmcv import Config
    baseline_cfg = Config.fromfile(str(baseline))
    prescf_cfg = Config.fromfile(str(prescf))
    assert 'temporal_agent_boxes' not in baseline_cfg.train_pipeline[-1]['keys']
    assert 'temporal_agent_boxes' in prescf_cfg.train_pipeline[-1]['keys']


def test_baseline_mode_numeric_identity_for_same_checkpoint_and_input():
    """The reference and explicitly-disabled PreSCF configs are identical.

    This exercises the actual detector's BaseLine future path rather than
    relying on source-string checks.  It is GPU-gated because OPUSHead's
    legacy voxel helpers allocate CUDA buffers during construction.
    """
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required for detector numeric identity test')
    from mmcv import Config
    from mmdet3d.models import build_model

    root = Path(__file__).parents[2]
    baseline_cfg = Config.fromfile(str(
        root / 'configs/sparseworld/nuscenes-temporal/'
        'sparseworld-traj-baseline.py'))
    prescf_cfg = Config.fromfile(str(
        root / 'configs/sparseworld/nuscenes-temporal/'
        'sparseworld-traj-prescf.py'))
    # Force the second config into the reference execution mode while
    # retaining its PreSCF data/config namespace.
    prescf_cfg.model.dsqe_mode = 'baseline'
    prescf_cfg.model.dsqe_cfg = dict(enabled=False, mode='baseline')

    device = torch.device('cuda')
    model_a = build_model(baseline_cfg.model,
                          train_cfg=baseline_cfg.get('train_cfg'),
                          test_cfg=baseline_cfg.get('test_cfg')).to(device).eval()
    num_queries = model_a.num_query + sum(model_a.num_fu_query)
    stamps = torch.cat([
        torch.zeros(model_a.num_query, dtype=torch.long),
        *[torch.full((count,), index, dtype=torch.long)
          for index, count in enumerate(model_a.num_fu_query, 1)]
    ]).to(device)
    model_a.pts_bbox_head.ind_stamps_all = stamps
    generator = torch.Generator(device=device).manual_seed(123)
    outs = dict(
        query_feat=torch.randn(1, num_queries, model_a.out_dim,
                               generator=generator, device=device),
        all_refine_pts=[torch.rand(
            1, num_queries, model_a.num_refines, 3,
            generator=generator, device=device)],
        all_cls_scores=[torch.randn(
            1, num_queries, model_a.num_refines, 17,
            generator=generator, device=device)])
    ego_feat = torch.randn(1, 1, model_a.out_dim,
                           generator=generator, device=device)
    with torch.no_grad():
        output_a = model_a._forward_baseline_scf(outs, ego_feat, [{}], {})
    expected = dict(
        cls_score=output_a['cls_score'].detach().cpu(),
        refine_pts=output_a['refine_pts'].detach().cpu(),
        forecast_points=[item.detach().cpu()
                         for item in output_a['forecast_points_list']],
        forecast_semantics=[item.detach().cpu()
                            for item in output_a['forecast_semantics_list']],
        pred_trajs=[item.detach().cpu()
                    for item in output_a['pred_trajs_list']])
    checkpoint = {key: value.detach().cpu().clone()
                  for key, value in model_a.state_dict().items()}
    del model_a
    torch.cuda.empty_cache()

    model_b = build_model(prescf_cfg.model,
                          train_cfg=prescf_cfg.get('train_cfg'),
                          test_cfg=prescf_cfg.get('test_cfg')).to(device).eval()
    load_result = model_b.load_state_dict(checkpoint, strict=True)
    assert not load_result.missing_keys and not load_result.unexpected_keys
    model_b.pts_bbox_head.ind_stamps_all = stamps
    outs_b = {key: [value.clone() for value in values]
              if isinstance(values, list) else values.clone()
              for key, values in outs.items()}
    with torch.no_grad():
        output_b = model_b._forward_baseline_scf(
            outs_b, ego_feat.clone(), [{}], {})
    assert torch.allclose(output_b['cls_score'].cpu(), expected['cls_score'],
                          atol=1e-6, rtol=1e-6)
    assert torch.allclose(output_b['refine_pts'].cpu(), expected['refine_pts'],
                          atol=1e-6, rtol=1e-6)
    for actual, reference in zip(output_b['forecast_points_list'],
                                 expected['forecast_points']):
        assert torch.allclose(actual.cpu(), reference, atol=1e-6, rtol=1e-6)
    for actual, reference in zip(output_b['forecast_semantics_list'],
                                 expected['forecast_semantics']):
        assert torch.allclose(actual.cpu(), reference, atol=1e-6, rtol=1e-6)
    for actual, reference in zip(output_b['pred_trajs_list'],
                                 expected['pred_trajs']):
        assert torch.allclose(actual.cpu(), reference, atol=1e-6, rtol=1e-6)


def test_prescf_source_is_recursive_and_has_no_residual_anchor():
    root = Path(__file__).parents[2]
    source = (root / 'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py').read_text()
    assert 'def _forward_prescf' in source
    assert 'def _forward_baseline_scf' in source
    assert 'base_points + residual' not in source
    assert '_baseline_future_step' not in source
    assert 'base_points' not in (root / 'mmdet3d/models/sparsedetectors/dsqe_dual_evolution.py').read_text()
    assert 'prescf_ego_pose_head' in source
    assert 'temporal_adjacent2ego' in source
    assert 'def _prescf_foreground_mask' in source
    # Future points are already in E_{t+1}; PreSCF must not route them through
    # the BaseLine temporal-trajectory mask a second time.
    prescf_body = source[source.index('def _forward_prescf'):]
    assert '_prescf_foreground_mask' in prescf_body


def test_prescf_foreground_mask_does_not_reapply_baseline_trajectory_warp():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    # Call the unbound helper so the test does not need to construct the full
    # detector.  A poisoned kwargs object proves the method is frame-local.
    points = torch.tensor([[[[0.2, 0.5, 0.5], [-0.1, 0.5, 0.5]]]])
    mask = SparseWorld4DTraj._prescf_foreground_mask(
        object(), points, interval=3, img_metas=None,
        kwargs={'temporal_trajs': object(), 'ego2lidar': object()})
    assert torch.equal(mask, torch.tensor([[[True, False]]]))


def test_prescf_training_pipeline_carries_ragged_actor_supervision():
    root = Path(__file__).parents[2]
    config = (root / 'configs/sparseworld/nuscenes-temporal/'
              'sparseworld-traj-finetune.py').read_text()
    assert 'temporal_agent_boxes' in config
    assert 'temporal_agent_feats' in config
    assert 'temporal_agent_labels' in config
    formatting = (root / 'mmdet3d/datasets/pipelines/formating.py').read_text()
    assert "DC(value, stack=False)" in formatting


def test_prescf_loss_keeps_dynamic_geometry_gradients():
    root = Path(__file__).parents[2]
    source = (root / 'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py').read_text()
    dynamic_call = source[source.index('self.pts_bbox_head._loss_dynamic'):]
    assert 'cache, refine_metric.detach()' not in dynamic_call
    head = (root / 'mmdet3d/models/sparsedetectors/opus_head.py').read_text()
    assert 'gt_distances, gt_to_pred = OPUSHead._chunked_nearest' in head


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
    query_role = torch.ones(1, num_carried + num_new, 1) * .5
    identity = warp.identity(1, feat.device, feat.dtype)
    result = module(feat, carried, new, torch.zeros(1, 1, 8), query_role,
                    identity, identity, warp)
    assert result['points'].shape == (1, num_carried + num_new, 4, 3)
    assert result['query_motion'].shape == (1, num_carried + num_new, 3)


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
    out = module(torch.zeros(1, 1, 4), points, points[:, :0],
                 torch.zeros(1, 1, 4), torch.ones(1, 1, 1),
                 identity, identity, warp)
    assert out['query_motion'].abs().sum() > 0
    assert decode_points(out['points'], pc_range).abs().sum() > 0


def test_dynamic_evolution_uses_prewarp_pool_and_preserves_xyz_delta():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.bbox.utils import decode_points, encode_points
    from mmdet3d.models.sparsedetectors.dsqe_dual_evolution import DSQEDualEvolution
    from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp

    pc_range = torch.tensor([-10., -10., -2., 10., 10., 2.])
    module = DSQEDualEvolution(4, 2, pc_range, motion_scale=1.)
    warp = DSQEEgoWarp(pc_range)
    points_metric = torch.tensor([[[[2., 0., 1.], [4., 0., 3.]]]])
    points = encode_points(points_metric, pc_range)
    # Make the point-level dynamic head emit a vertical correction.  The
    # default path must preserve xyz; only explicit planar_motion_only may
    # zero z.
    with torch.no_grad():
        module.dynamic_head[-1].bias.fill_(0.)
        module.dynamic_head[-1].bias[2::3].fill_(0.5)
    next_to_current = torch.eye(4).unsqueeze(0)
    next_to_current[:, 0, 3] = 1.
    out = module(torch.zeros(1, 1, 4), points, points[:, :0],
                 torch.zeros(1, 1, 4), torch.ones(1, 1, 1),
                 next_to_current, next_to_current, warp)
    evolved = decode_points(out['dynamic_points_metric'], pc_range)
    # ego warp maps current points to x-1 in the next frame; the learned
    # z correction remains visible and is not forcibly clamped to zero.
    assert torch.all(evolved[..., 2] > 1.)


def test_cumulative_pose_targets_are_converted_to_adjacent_transforms():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    def transform(x, yaw):
        matrix = torch.eye(4)
        c, s = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
        matrix[:2, :2] = torch.tensor([[c, -s], [s, c]])
        matrix[0, 3] = x
        return matrix

    first = transform(1.0, 0.1)
    second = transform(2.5, 0.3)
    targets = SparseWorld4DTraj._build_adjacent_ego_targets(
        {'temporal2ego': {0: first[None], 1: second[None]}},
        batch_size=1, num_steps=2, device=torch.device('cpu'),
        dtype=torch.float32)
    assert torch.allclose(targets[0], first[None], atol=1e-6)
    assert torch.allclose(targets[1],
                          torch.linalg.inv(first)[None] @ second[None],
                          atol=1e-6)
    assert torch.allclose(targets[0] @ targets[1], second[None], atol=1e-6)


def test_ego_warp_rotation_uses_column_transform_convention():
    """A +90-degree ego transform must map +x to +y, not -y."""
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp

    warp = DSQEEgoWarp([-10., -10., -2., 10., 10., 2.])
    points = torch.tensor([[[[1., 0., 0.]]]])
    pose = torch.tensor([[0., 0., 1., 0.]])  # +90 degrees
    transformed = warp.transform_metric(points, warp.pose_to_matrix(pose))
    assert torch.allclose(transformed, torch.tensor([[[[0., 1., 0.]]]]),
                          atol=1e-6)


def test_role_assignment_rejects_others_and_keeps_dynamic_class_ids():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.opus_head import OPUSHead

    points = torch.tensor([[0., 0., 0.], [0., 0., 0.], [8., 8., 0.]])
    labels = torch.tensor([0, 4, 11])
    metadata = dict(centers=torch.tensor([[0., 0., 0.]]),
                    radius=torch.tensor([2.]), role=torch.tensor([1.]),
                    valid=torch.tensor([True]), dims=torch.tensor([[4., 2., 2.]]),
                    yaw=torch.tensor([0.]))
    target, valid = OPUSHead._assign_motion_state_roles(
        points, labels, metadata,
        static_ids=[1, 8, 11, 12, 13, 14, 15, 16],
        dynamic_ids=[2, 3, 4, 5, 6, 7, 9, 10])
    assert not valid[0]  # class-0 ``others`` is ignored
    assert valid[1] and target[1] == 1
    assert valid[2] and target[2] == 0


def test_bidirectional_dynamic_loss_backpropagates_to_predicted_points():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.opus_head import OPUSHead

    fake = SimpleNamespace(
        _chunked_nearest_indices=OPUSHead._chunked_nearest_indices)
    predicted = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]],
                             requires_grad=True)
    output = {'role_pred': torch.full((1, 1, 2, 1), 0.8)}
    cache = dict(gt_points_list=[torch.tensor([[2., 0., 0.], [3., 0., 0.]])],
                 gt_role_target_list=[torch.ones(2)],
                 gt_role_valid_list=[torch.ones(2, dtype=torch.bool)])
    loss = OPUSHead._loss_dynamic(
        fake, torch.zeros(1, 1, 2, 17), output, cache, predicted)
    loss.backward()
    assert loss.item() > 0
    assert predicted.grad is not None and predicted.grad.abs().sum() > 0


def _dynamic_loss_fixture(torch):
    from mmdet3d.models.sparsedetectors.opus_head import OPUSHead
    fake = SimpleNamespace(
        dsqe_cfg=dict(
            role_match_max_distance=2.5, dynamic_huber_beta=0.2,
            dynamic_semantic_weight=0.,
            dynamic_class_ids=[2, 3, 4, 5, 6, 7, 9, 10],
            static_class_ids=[1, 8, 11, 12, 13, 14, 15, 16]),
        _chunked_nearest_indices=OPUSHead._chunked_nearest_indices)
    points = torch.tensor(
        [[[-5., 0., 0.], [1., 0., 0.], [8., 0., 0.]]],
        requires_grad=True)
    cache = dict(
        gt_points_list=[torch.tensor([[2., 0., 0.]])],
        gt_role_target_list=[torch.ones(1)],
        gt_role_valid_list=[torch.ones(1, dtype=torch.bool)],
        role_target=torch.tensor([[[0., 1., 0.]]]),
        role_valid=torch.ones(1, 1, 3, dtype=torch.bool))
    scores = torch.zeros(1, 1, 3, 17)
    output = dict(role_pred=torch.zeros(1, 1, 3, 1))
    return fake, points, scores, output, cache


def test_dynamic_pred_to_gt_excludes_static_predictions():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.opus_head import OPUSHead

    fake, points, scores, output, cache = _dynamic_loss_fixture(torch)
    dynamic = OPUSHead._loss_dynamic(
        fake, scores, output, cache, points, return_diagnostics=True)
    dynamic['loss_dynamic_pred_to_gt'].backward()
    # Only the GT-confirmed dynamic middle point participates in this
    # direction. Static predictions remain exactly gradient-free.
    assert points.grad[0, 1].abs().sum() > 0
    assert points.grad[0, 0].abs().sum() == 0
    assert points.grad[0, 2].abs().sum() == 0
    assert dynamic['dynamic_pred_valid_ratio'] == pytest.approx(1 / 3)


def test_dynamic_gt_to_pred_survives_role_collapse():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.opus_head import OPUSHead

    fake, points, scores, output, cache = _dynamic_loss_fixture(torch)
    output['role_pred'].zero_()  # predicted rho collapses to static
    dynamic = OPUSHead._loss_dynamic(
        fake, scores, output, cache, points, return_diagnostics=True)
    dynamic['loss_dynamic_gt_to_pred'].backward()
    assert dynamic['loss_dynamic_gt_to_pred'] > 0
    assert points.grad is not None and points.grad.abs().sum() > 0
    assert torch.isfinite(points.grad).all()


def test_role_loss_is_class_balanced_focal_and_reports_f1():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.opus_head import OPUSHead

    fake = SimpleNamespace(
        dsqe_cfg=dict(role_dynamic_weight='auto',
                      role_dynamic_weight_max=20., role_focal_gamma=2.),
        _masked_mean=OPUSHead._masked_mean)
    logits = torch.zeros(1, 1, 4, 1, requires_grad=True)
    output = dict(
        role_logits=logits,
        pool_weights=torch.full((1, 1, 4, 1), 0.25),
        query_role=torch.full((1, 1, 1), 0.5, requires_grad=True))
    cache = dict(
        role_target=torch.tensor([[[1., 0., 0., 0.]]]),
        role_valid=torch.ones(1, 1, 4, dtype=torch.bool))
    loss = OPUSHead._loss_role(fake, output, cache)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum() > 0
    metrics = OPUSHead._role_metrics(output, cache)
    assert {
        'role_precision', 'role_recall', 'role_f1', 'query_precision',
        'query_recall', 'query_f1', 'dynamic_valid_ratio', 'ignored_ratio',
        'unmatched_ratio'} <= set(metrics)
    assert metrics['dynamic_valid_ratio'] == 0.25


def test_role_loss_uses_current_state_cache():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.opus_head import OPUSHead
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    current = dict(
        role_target=torch.tensor([[[1., 0.]]]),
        role_valid=torch.tensor([[[True, True]]]))
    future = dict(
        role_target=torch.tensor([[[0., 1.]]]),
        role_valid=torch.tensor([[[False, True]]]))
    state = dict(current_role_cache=current, future_match_cache=future)
    selected_current, selected_future = \
        SparseWorld4DTraj._prescf_supervision_caches(state)
    assert selected_current is current and selected_future is future
    fake = SimpleNamespace(
        dsqe_cfg=dict(role_dynamic_weight=1., role_focal_gamma=2.),
        _masked_mean=OPUSHead._masked_mean)
    output = dict(role_logits=torch.zeros(1, 1, 2, 1))
    expected = OPUSHead._loss_role(fake, output, current)
    actual = OPUSHead._loss_role(fake, output, selected_current)
    assert torch.equal(actual, expected)


def test_future_point_perturbation_does_not_change_current_role_target():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    current_target = torch.tensor([[[1., 0., 1.]]])
    current_valid = torch.tensor([[[True, True, False]]])
    current = dict(role_target=current_target, role_valid=current_valid)
    state_a = dict(
        current_role_cache=current,
        future_match_cache=dict(
            role_target=torch.zeros_like(current_target),
            role_valid=torch.zeros_like(current_valid)))
    state_b = dict(
        current_role_cache=current,
        future_match_cache=dict(
            role_target=torch.ones_like(current_target),
            role_valid=torch.ones_like(current_valid)))
    cache_a, _ = SparseWorld4DTraj._prescf_supervision_caches(state_a)
    cache_b, _ = SparseWorld4DTraj._prescf_supervision_caches(state_b)
    assert torch.equal(cache_a['role_target'], cache_b['role_target'])
    assert torch.equal(cache_a['role_valid'], cache_b['role_valid'])


def test_unmatched_ratio_uses_unique_thresholded_gt_coverage():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.opus_head import OPUSHead

    output = dict(role_logits=torch.zeros(1, 1, 4, 1))
    cache = dict(
        role_target=torch.tensor([[[1., 1., 0., 0.]]]),
        role_valid=torch.ones(1, 1, 4, dtype=torch.bool),
        gt_role_target_list=[torch.tensor([1., 1., 0.])],
        gt_role_valid_list=[torch.tensor([True, True, True])],
        # Many prediction->GT assignments to GT 0 are deliberately present;
        # they must not count as coverage for GT 1 or GT 2.
        pred_paired_idx_list=[torch.tensor([0, 0, 0, 0])],
        gt_to_pred_distance_list=[torch.tensor([0.1, 3.0, 4.0])],
        role_match_max_distance=2.5)
    metrics = OPUSHead._role_metrics(output, cache)
    assert metrics['unique_gt_coverage_ratio'] == pytest.approx(1 / 3)
    assert metrics['unmatched_ratio'] == pytest.approx(2 / 3)
    assert metrics['dynamic_gt_covered_ratio'] == pytest.approx(1 / 2)
    assert metrics['dynamic_unmatched_ratio'] == pytest.approx(1 / 2)
    assert metrics['static_unmatched_ratio'] == pytest.approx(1.0)

    empty = dict(
        role_target=torch.zeros(1, 1, 1),
        role_valid=torch.zeros(1, 1, 1, dtype=torch.bool),
        gt_role_target_list=[torch.zeros(0)],
        gt_role_valid_list=[torch.zeros(0, dtype=torch.bool)],
        gt_to_pred_distance_list=[torch.zeros(0)],
        role_match_max_distance=2.5)
    empty_metrics = OPUSHead._role_metrics(
        dict(role_logits=torch.zeros(1, 1, 1, 1)), empty)
    for name in ('unmatched_ratio', 'dynamic_unmatched_ratio',
                 'static_unmatched_ratio', 'dynamic_gt_covered_ratio',
                 'unique_gt_coverage_ratio'):
        assert torch.isfinite(empty_metrics[name])
        assert empty_metrics[name] == 0


def test_local_attention_uses_only_nearest_k_senders():
    torch = pytest.importorskip('torch')
    from torch import nn
    from mmdet3d.models.sparsedetectors.dsqe_dual_interaction import DSQEDualInteraction

    class MeanAttention(nn.Module):
        def forward(self, query, key, value, **kwargs):
            return value.mean(1, keepdim=True).expand_as(query), None

    block = DSQEDualInteraction(4, num_heads=1, local_k=1, dropout=0.)
    query = torch.zeros(1, 1, 4)
    key = torch.zeros(1, 3, 4)
    value = torch.tensor([[[1., 1., 1., 1.],
                           [10., 10., 10., 10.],
                           [100., 100., 100., 100.]]])
    query_centers = torch.tensor([[[0., 0., 0.]]])
    key_centers = torch.tensor([[[0.1, 0., 0.], [2., 0., 0.], [5., 0., 0.]]])
    result = block._local_attention(
        MeanAttention(), query, key, value, query_centers, key_centers,
        torch.ones(1, 3, 1))
    assert torch.allclose(result, torch.ones_like(result))

    empty = block._local_attention(
        MeanAttention(), query, key, value, query_centers, key_centers,
        torch.zeros(1, 3, 1))
    assert torch.equal(empty, torch.zeros_like(empty))


def test_ragged_actor_fields_are_not_stacked_by_pipeline():
    torch = pytest.importorskip('torch')
    from mmdet3d.datasets.pipelines.formating import DefaultFormatBundle3D

    result = DefaultFormatBundle3D(class_names=[])(dict(
        temporal_agent_boxes=torch.zeros(3, 9),
        temporal_agent_feats=torch.zeros(3, 33),
        temporal_agent_labels=torch.zeros(3, dtype=torch.long)))
    for key in ('temporal_agent_boxes', 'temporal_agent_feats',
                'temporal_agent_labels'):
        assert result[key].stack is False


def test_collated_ragged_actor_fields_round_trip_per_sample():
    torch = pytest.importorskip('torch')
    from mmcv.parallel import DataContainer, collate
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    collated = collate([
        {'actors': DataContainer(torch.zeros(3, 9), stack=False)},
        {'actors': DataContainer(torch.zeros(5, 9), stack=False)},
    ], samples_per_gpu=2)['actors']
    # MMCV's actual collated representation is intentionally nested.
    result = SparseWorld4DTraj._to_batch_sequence(
        collated, torch.device('cpu'), torch.float32)
    assert [tuple(item.shape) for item in result] == [(3, 9), (5, 9)]

    collated_labels = collate([
        {'labels': DataContainer(torch.arange(3), stack=False)},
        {'labels': DataContainer(torch.arange(5), stack=False)},
    ], samples_per_gpu=2)['labels']
    labels = SparseWorld4DTraj._to_batch_sequence(
        collated_labels, torch.device('cpu'), torch.long)
    assert [tuple(item.shape) for item in labels] == [(3,), (5,)]


def test_collated_batch2_actor_metadata_keeps_sample_alignment():
    torch = pytest.importorskip('torch')
    from mmcv.parallel import DataContainer, collate
    from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    samples = []
    for count, label, shift in ((2, 4, 0.), (3, 1, 10.)):
        boxes = torch.zeros(count, 9)
        boxes[:, 0] = torch.arange(count) + shift
        boxes[:, 3:6] = torch.tensor([2., 4., 2.])
        feats = torch.zeros(count, 4)
        feats[:, 2] = 1
        samples.append(dict(
            boxes=DataContainer(boxes, stack=False),
            feats=DataContainer(feats, stack=False),
            labels=DataContainer(torch.full((count,), label), stack=False),
            lidar2ego=torch.eye(4)))
    batch = collate(samples, samples_per_gpu=2)
    fake = SimpleNamespace(
        dsqe_cfg=dict(role_box_inflation=0., role_box_dims_order='wlh',
                      role_trajectory_mode='increment'),
        dynamic_class_ids=(2, 3, 4, 5, 6, 7, 9, 10), num_fu_frames=1,
        ego_warp=DSQEEgoWarp([-40., -40., -1., 40., 40., 5.4]),
        _to_batch_sequence=SparseWorld4DTraj._to_batch_sequence,
        _build_adjacent_ego_targets=lambda kwargs, size, steps, device, dtype:
            SparseWorld4DTraj._build_adjacent_ego_targets(
                kwargs, size, steps, device, dtype))
    metadata = SparseWorld4DTraj._build_role_metadata(
        fake, dict(
            temporal_agent_boxes=batch['boxes'],
            temporal_agent_feats=batch['feats'],
            temporal_agent_labels=batch['labels'],
            temporal_agent_lidar2ego=batch['lidar2ego'],
            temporal_adjacent2ego={0: torch.eye(4).repeat(2, 1, 1)}),
        interval=1)
    assert [item['centers'].shape[0] for item in metadata] == [2, 3]
    assert metadata[0]['role'].eq(1).all()  # occupancy car id 4
    assert metadata[1]['role'].eq(0).all()  # occupancy barrier id 1
    assert torch.allclose(metadata[0]['dims'][0], torch.tensor([2., 4., 2.]))


def test_lidar_actor_calibration_is_applied_before_role_matching():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    fake = SimpleNamespace(
        dsqe_cfg=dict(role_box_inflation=0.0,
                      role_trajectory_mode='increment'),
        dynamic_class_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        num_fu_frames=1,
        ego_warp=DSQEEgoWarp([-40., -40., -1., 40., 40., 5.4]),
        _to_batch_sequence=SparseWorld4DTraj._to_batch_sequence,
        _build_adjacent_ego_targets=lambda kwargs, batch, steps, device, dtype:
            SparseWorld4DTraj._build_adjacent_ego_targets(
                kwargs, batch, steps, device, dtype))
    boxes = torch.tensor([[1., 0., 0., 2., 4., 2., 0., 0., 0.]])
    feats = torch.zeros(1, 4)
    feats[0, 2] = 1.0
    l2e = torch.eye(4)
    l2e[0, 3] = 2.0
    metadata = SparseWorld4DTraj._build_role_metadata(
        fake,
        dict(temporal_agent_boxes=boxes,
             temporal_agent_feats=feats,
             temporal_agent_labels=torch.tensor([4]),
             temporal_agent_lidar2ego=l2e,
             temporal_adjacent2ego={0: torch.eye(4)}),
        interval=0)[0]
    assert torch.allclose(metadata['centers'][0, 0], torch.tensor(3.))


def test_actor_absolute_compatibility_mode_and_yaw_convention():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    fake = SimpleNamespace(
        dsqe_cfg=dict(role_box_inflation=0.5,
                      role_trajectory_mode='absolute'),
        dynamic_class_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        num_fu_frames=6,
        ego_warp=DSQEEgoWarp([-40., -40., -1., 40., 40., 5.4]),
        _to_batch_sequence=SparseWorld4DTraj._to_batch_sequence,
        _build_adjacent_ego_targets=lambda kwargs, batch, steps, device, dtype:
            SparseWorld4DTraj._build_adjacent_ego_targets(
                kwargs, batch, steps, device, dtype))
    boxes = torch.tensor([[0., 0., 0., 4., 2., 2., 0., 0., 0.]])
    feats = torch.zeros(1, 34)
    # Compatibility mode interprets the first two xy pairs as horizon
    # displacements.  VAD production caches use increment mode below.
    feats[0, :4] = torch.tensor([1., 0., 2., 0.])
    feats[0, 12:18] = 1  # future-valid mask
    feats[0, -6:-4] = torch.tensor([0.1, 0.2])
    first = torch.eye(4); first[0, 3] = 1
    second = torch.eye(4); second[0, 3] = 1
    metadata = SparseWorld4DTraj._build_role_metadata(
        fake,
        dict(temporal_agent_boxes=boxes,
             temporal_agent_feats=feats,
             temporal_agent_labels=torch.tensor([4]),
             temporal_adjacent2ego={0: first, 1: second}),
        interval=2)[0]
    # Actor is at +2m in E0 while E2's origin moves +2m: x(E2) == 0m.
    assert torch.allclose(metadata['centers'][0, :2],
                          torch.tensor([0., 0.]), atol=1e-6)
    # gt_boxes uses SECOND yaw while the cache stores raw yaw deltas; the
    # conversion therefore subtracts the raw +0.2 delta.
    assert torch.allclose(metadata['yaw'][0], torch.tensor(-0.2), atol=1e-6)
    assert metadata['valid'][0] and metadata['role'][0] == 1


def test_actor_increment_cache_can_be_selected_explicitly():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    fake = SimpleNamespace(
        dsqe_cfg=dict(role_box_inflation=0.0,
                      role_trajectory_mode='increment'),
        dynamic_class_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        num_fu_frames=6,
        ego_warp=DSQEEgoWarp([-40., -40., -1., 40., 40., 5.4]),
        _to_batch_sequence=SparseWorld4DTraj._to_batch_sequence,
        _build_adjacent_ego_targets=lambda kwargs, batch, steps, device, dtype:
            SparseWorld4DTraj._build_adjacent_ego_targets(
                kwargs, batch, steps, device, dtype))
    boxes = torch.tensor([[0., 0., 0., 4., 2., 2., 0., 0., 0.]])
    feats = torch.zeros(1, 34)
    feats[0, :4] = torch.tensor([1., 0., 2., 0.])
    feats[0, 12:18] = 1
    feats[0, -6:-4] = torch.tensor([0.1, 0.2])
    first = torch.eye(4); first[0, 3] = 1
    second = torch.eye(4); second[0, 3] = 1
    metadata = SparseWorld4DTraj._build_role_metadata(
        fake,
        dict(temporal_agent_boxes=boxes,
             temporal_agent_feats=feats,
             temporal_agent_labels=torch.tensor([4]),
             temporal_adjacent2ego={0: first, 1: second}),
        interval=2)[0]
    # Explicit increment mode integrates +1 + +2 = +3m in E0.
    assert torch.allclose(metadata['centers'][0, :2],
                          torch.tensor([1., 0.]), atol=1e-6)
    assert torch.allclose(metadata['yaw'][0], torch.tensor(-0.3), atol=1e-6)


def test_actor_yaw_applies_lidar_and_future_ego_rotation_in_second_space():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    fake = SimpleNamespace(
        dsqe_cfg=dict(role_box_inflation=0.0,
                      role_box_dims_order='wlh',
                      role_yaw_encoding='raw',
                      role_trajectory_mode='increment'),
        dynamic_class_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        num_fu_frames=1,
        ego_warp=DSQEEgoWarp([-40., -40., -1., 40., 40., 5.4]),
        _to_batch_sequence=SparseWorld4DTraj._to_batch_sequence,
        _build_adjacent_ego_targets=lambda kwargs, batch, steps, device, dtype:
            SparseWorld4DTraj._build_adjacent_ego_targets(
                kwargs, batch, steps, device, dtype))
    boxes = torch.tensor([[0., 0., 0., 4., 2., 2., 0., 0., 0.]])
    feats = torch.zeros(1, 4)
    feats[0, 2] = 1.0
    quarter_turn = torch.tensor(torch.pi / 2)
    c, s = torch.cos(quarter_turn), torch.sin(quarter_turn)
    lidar2ego = torch.eye(4)
    lidar2ego[:2, :2] = torch.tensor([[c, -s], [s, c]])
    e0_to_e1 = torch.eye(4)
    e0_to_e1[:2, :2] = torch.tensor([[c, -s], [s, c]])
    metadata = SparseWorld4DTraj._build_role_metadata(
        fake,
        dict(temporal_agent_boxes=boxes,
             temporal_agent_feats=feats,
             temporal_agent_labels=torch.tensor([4]),
             temporal_agent_lidar2ego=lidar2ego,
             temporal_adjacent2ego={0: torch.linalg.inv(e0_to_e1)}),
        interval=1)[0]
    # SECOND yaw rotates with the opposite sign of the raw frame rotation:
    # -lidar_yaw - actor_delta - (E0->E1)_yaw = -pi.
    assert torch.allclose(metadata['yaw'][0], torch.tensor(-torch.pi),
                          atol=1e-5)


def test_actor_box_dimensions_preserve_vad_wlh_for_second_yaw_footprint():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp
    from mmdet3d.models.sparsedetectors.opus_head import OPUSHead
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    fake = SimpleNamespace(
        dsqe_cfg=dict(role_box_inflation=0.0, role_box_dims_order='wlh'),
        dynamic_class_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        num_fu_frames=1,
        ego_warp=DSQEEgoWarp([-40., -40., -1., 40., 40., 5.4]),
        _to_batch_sequence=SparseWorld4DTraj._to_batch_sequence,
        _build_adjacent_ego_targets=lambda kwargs, batch, steps, device, dtype:
            SparseWorld4DTraj._build_adjacent_ego_targets(
                kwargs, batch, steps, device, dtype))
    # VAD concatenates native nuScenes Box.wlh into ``gt_boxes``.
    boxes = torch.tensor([[0., 0., 0., 2., 4., 2., 0., 0., 0.]])
    feats = torch.zeros(1, 2 + 1 + 1)
    feats[0, 2] = 1.0  # one valid future step
    metadata = SparseWorld4DTraj._build_role_metadata(
        fake,
        dict(temporal_agent_boxes=boxes,
             temporal_agent_feats=feats,
             temporal_agent_labels=torch.tensor([4]),
             temporal_adjacent2ego={0: torch.eye(4)}),
        interval=1)[0]
    assert torch.allclose(metadata['dims'][0], torch.tensor([2., 4., 2.]))
    # With SECOND yaw=0, local x uses width (half extent 1) and local y uses
    # length (half extent 2).  This detects an accidental axis-only swap.
    _, valid = OPUSHead._assign_motion_state_roles(
        torch.tensor([[1.5, 0., 0.], [0., 1.5, 0.]]),
        torch.tensor([4, 4]), metadata,
        static_ids=[1, 8, 11, 12, 13, 14, 15, 16],
        dynamic_ids=[2, 3, 4, 5, 6, 7, 9, 10])
    assert torch.equal(valid, torch.tensor([False, True]))


def _association_metadata(torch, valid=(True, True)):
    return [dict(
        centers=torch.tensor([[0., 0., 0.], [10., 0., 0.]]),
        radius=torch.tensor([3., 3.]),
        role=torch.tensor([1., 1.]),
        valid=torch.tensor(valid),
        labels=torch.tensor([4, 4]),
        dims=torch.tensor([[4., 4., 2.], [4., 4., 2.]]),
        yaw=torch.tensor([0., 0.]),
        inflation=0.)]


def test_carried_actor_association_is_propagated():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    points = torch.tensor([[[[0., 0., 0.], [1., 0., 0.]]]])
    association = SparseWorld4DTraj._initialize_actor_association(
        points, _association_metadata(torch))
    point_actor_id = association['point_actor_id']
    assert point_actor_id.tolist() == [[[0, 0]]]
    # Six arbitrarily drifted future geometries do not rewrite identity.
    for _ in range(6):
        drifted_points = points + torch.randn_like(points) * 100
        del drifted_points  # geometry is intentionally irrelevant after activation
        valid = SparseWorld4DTraj._refresh_actor_association(
            point_actor_id, _association_metadata(torch))
        assert point_actor_id.tolist() == [[[0, 0]]]
        assert valid.all()
        assert SparseWorld4DTraj._query_actor_ids(
            point_actor_id).item() == 0


def test_new_query_actor_association_is_initialized_independently():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    metadata = _association_metadata(torch)
    carried = SparseWorld4DTraj._initialize_actor_association(
        torch.tensor([[[[0., 0., 0.], [1., 0., 0.]]]]), metadata)
    new = SparseWorld4DTraj._initialize_actor_association(
        torch.tensor([[[[10., 0., 0.], [11., 0., 0.]]]]), metadata)
    combined = torch.cat(
        [carried['point_actor_id'], new['point_actor_id']], dim=1)
    assert combined.tolist() == [[[0, 0], [1, 1]]]
    assert SparseWorld4DTraj._query_actor_ids(combined).tolist() == [[0, 1]]


def test_actor_invalidity_masks_future_role_supervision():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    points = torch.tensor([[[[0., 0., 0.], [1., 0., 0.]]]])
    association = SparseWorld4DTraj._initialize_actor_association(
        points, _association_metadata(torch))
    point_actor_id = association['point_actor_id']
    original = dict(
        role_target=torch.zeros(1, 1, 2),
        role_valid=torch.ones(1, 1, 2, dtype=torch.bool))
    future = SparseWorld4DTraj._apply_actor_association_to_role_cache(
        original, point_actor_id, _association_metadata(torch, (False, True)))
    assert future['point_actor_id'].tolist() == [[[0, 0]]]
    assert future['query_actor_id'].item() == 0  # identity persists
    assert future['role_target'].eq(1).all()
    assert not future['role_valid'].any()  # supervision follows future validity
    assert not future['actor_valid'].any()


def test_shared_ego_pose_is_exported_in_vad_lidar_planning_frame():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp

    warp = DSQEEgoWarp([-40., -40., -1., 40., 40., 5.4])
    cumulative = torch.eye(4).unsqueeze(0)
    cumulative[:, 0, 3] = 3.0  # three metres forward in E0 ego
    # nuScenes LIDAR_TOP is approximately -90 degrees from ego axes.
    lidar_to_ego = torch.eye(4).unsqueeze(0)
    angle = torch.tensor(-torch.pi / 2)
    c, s = angle.cos(), angle.sin()
    lidar_to_ego[:, :2, :2] = torch.tensor([[c, -s], [s, c]])
    lidar_cumulative = warp.ego_cumulative_to_lidar(
        cumulative, lidar_to_ego)
    # The same pose state becomes +y in the fixed current-LiDAR frame.
    assert torch.allclose(lidar_cumulative[0, :2, 3],
                          torch.tensor([0., 3.]), atol=1e-6)


def test_joint_refine_spatial_attention_runs_over_shared_queries():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.dsqe_joint_refine import DSQEJointRefine

    module = DSQEJointRefine(8, 4, num_classes=17, num_heads=2)
    seen = []
    handle = module.spatial_attn[0].register_forward_pre_hook(
        lambda _module, args: seen.append(tuple(args[0].shape)))
    feature = torch.randn(1, 3, 8)
    points = torch.randn(1, 3, 4, 3)
    output = module(feature, feature, feature, points)
    handle.remove()
    # Three flattened Query neighborhoods, each with one receiver token.
    assert seen == [(3, 1, 8)]
    assert output['point_correction'].shape == (1, 3, 4, 3)


def test_absolute_semantic_head_warm_start_is_function_equivalent():
    torch = pytest.importorskip('torch')
    from torch import nn
    from mmdet3d.models.sparsedetectors.dsqe_joint_refine import DSQEJointRefine

    torch.manual_seed(7)
    baseline = nn.Sequential(
        nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(),
        nn.Linear(8, 4 * 3))
    prescf = DSQEJointRefine(8, 4, num_classes=3, num_heads=1)
    with torch.no_grad():
        for index in (0, 2, 4):
            prescf.semantic_head[index].load_state_dict(
                baseline[index].state_dict())
        prescf.point_adapter_gate.fill_(1.)
    feature = torch.randn(2, 3, 8)
    points = torch.randn(2, 3, 4, 3)
    expected = baseline(feature).reshape(2, 3, 4, 3)
    actual = prescf.predict_semantics(
        feature, points, use_point_adapter=False)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_absolute_semantic_head_point_adapter_is_coordinate_sensitive():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.dsqe_joint_refine import DSQEJointRefine

    module = DSQEJointRefine(8, 4, num_classes=3, num_heads=1)
    with torch.no_grad():
        module.point_adapter_gate.fill_(1.)
        module.point_adapter[-1].weight.fill_(1.)
        module.point_adapter[-1].bias.zero_()
    feature = torch.zeros(1, 1, 8)
    points_a = torch.zeros(1, 1, 4, 3)
    points_b = points_a.clone()
    points_b[..., 0] = 1.
    logits_a = module.predict_semantics(feature, points_a)
    logits_b = module.predict_semantics(feature, points_b)
    assert not torch.allclose(logits_a, logits_b)


def test_role_knn_has_distance_rejection_and_reports_both_directions():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.opus_head import OPUSHead

    distances, indices = OPUSHead._chunked_nearest(
        torch.tensor([[0., 0., 0.], [10., 0., 0.]]),
        torch.tensor([[0., 0., 0.]]))
    assert indices.tolist() == [0, 0]
    assert distances.tolist() == [0., 10.]
    # A 2.5m matching radius keeps the nearby target and ignores the far one.
    valid = distances <= 2.5
    assert valid.tolist() == [True, False]


def test_future_role_cache_uses_prediction_to_gt_distance_direction():
    """Each predicted point must be filtered by its own nearest-GT distance.

    ``_get_target_single`` returns ``pred_index`` as GT indices (one value per
    predicted point).  Re-indexing the *distance* tensor with those values
    would accidentally reuse the first prediction's distance whenever several
    predictions share a GT voxel.
    """
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.bbox.utils import encode_points
    from mmdet3d.models.sparsedetectors.opus_head import OPUSHead

    predicted_metric = torch.tensor([[[0., 0., 0.], [10., 0., 0.]]])
    refine_pts = encode_points(predicted_metric, torch.tensor(
        [-20., -20., -2., 20., 20., 2.])).reshape(1, 1, 2, 3)
    gt_points = torch.tensor([[0., 0., 0.], [100., 0., 0.]])
    gt_labels = torch.tensor([4, 4])
    motion = dict(
        centers=torch.tensor([[0., 0., 0.]]),
        radius=torch.tensor([2.]), role=torch.tensor([1.]),
        valid=torch.tensor([True]),
        dims=torch.tensor([[4., 4., 2.]]), yaw=torch.tensor([0.]))
    fake = SimpleNamespace(
        pc_range=torch.tensor([-20., -20., -2., 20., 20., 2.]),
        num_classes=17,
        dsqe_cfg=dict(role_match_max_distance=2.5),
        get_sparse_voxels=lambda _voxel: ([gt_points], [gt_labels]),
        _assign_motion_state_roles=OPUSHead._assign_motion_state_roles,
        _chunked_nearest=OPUSHead._chunked_nearest,
        _get_target_single=lambda _pred, _gt, _labels: (
            torch.tensor([4, 4]), torch.tensor([], dtype=torch.long),
            torch.tensor([0, 0]), torch.ones(2, 17), torch.ones(0)))
    cache = OPUSHead.build_future_match_cache(
        fake, refine_pts, torch.empty(1), role_metadata=[motion])
    assert cache['role_target'].shape == (1, 1, 2)
    assert cache['role_valid'].tolist() == [[[True, False]]]


def test_dual_interaction_loads_legacy_gate_checkpoint():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.dsqe_dual_interaction import DSQEDualInteraction

    source = DSQEDualInteraction(8, num_heads=2, dropout=0.)
    legacy = source.state_dict()
    legacy.pop('lambda_ds_raw')
    legacy.pop('lambda_sd_ratio_raw')
    legacy['dynamic_from_static'] = torch.tensor(2.)
    legacy['static_from_dynamic'] = torch.tensor(0.5)
    target = DSQEDualInteraction(8, num_heads=2, dropout=0.)
    target.load_state_dict(legacy, strict=True)
    assert torch.allclose(target.dynamic_from_static_gate, torch.tensor(2.))
    assert torch.allclose(target.static_from_dynamic_gate, torch.tensor(0.5))


def test_staged_tass_gradient_gate_is_ddp_safe():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    fake = SimpleNamespace(_prescf_tass_frozen=True)
    gradient = torch.ones(3)
    assert torch.equal(SparseWorld4DTraj._prescf_tass_gradient_gate(
        fake, gradient), torch.zeros(3))
    fake._prescf_tass_frozen = False
    assert torch.equal(SparseWorld4DTraj._prescf_tass_gradient_gate(
        fake, gradient), gradient)


def test_interaction_and_joint_stage_gates_ramp_from_identity():
    pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    fake = SimpleNamespace(
        training=True, curr_epoch=0, prescf_stage1_end_epoch=5,
        prescf_stage2_end_epoch=16, prescf_interaction_ramp_epochs=4,
        prescf_joint_ramp_epochs=4)
    first = SparseWorld4DTraj._prescf_stage_gates(fake)
    assert first[0] > 0 and first[1] > 0
    fake.curr_epoch = 5
    assert SparseWorld4DTraj._prescf_stage_gates(fake)[0] > first[0]
    fake.curr_epoch = 16
    assert SparseWorld4DTraj._prescf_stage_gates(fake)[0] == 1.
    assert SparseWorld4DTraj._prescf_stage_gates(fake)[1] > first[1]
    fake.training = False
    assert SparseWorld4DTraj._prescf_stage_gates(fake) == (1., 1.)


def test_stage1_gate_and_curriculum_contract_are_nonzero_and_explicit():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj

    fake = SimpleNamespace(
        training=True, curr_epoch=0, dsqe_mode='prescf', num_fu_frames=6,
        dsqe_cfg=dict(stage1_end_epoch=5, stage2_end_epoch=16),
        prescf_stage1_end_epoch=5, prescf_stage2_end_epoch=16,
        prescf_forecast_curriculum=(1, 2, 3, 6))
    for epoch, expected in ((0, 1), (5, 2), (9, 3), (13, 6)):
        fake.curr_epoch = epoch
        assert SparseWorld4DTraj._num_forecast_frames(fake) == expected


def _build_cuda_prescf_detector(torch):
    from mmcv import Config
    from mmdet3d.models import build_model

    root = Path(__file__).parents[2]
    cfg = Config.fromfile(str(
        root / 'configs/sparseworld/nuscenes-temporal/'
        'sparseworld-traj-prescf.py'))
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg')).cuda()
    return model


def test_all_prescf_modules_receive_effective_gradients():
    """Use real detector losses, including ragged actor supervision.

    Two optimizer iterations are intentional: zero-initialized point/semantic
    adapters expose their terminal projection on iteration one and their
    preceding layers after that projection has moved away from zero.
    """
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required for real forward_train gradient test')
    from tools.test_prescf_ddp_smoke import (
        _gradient_stats, _group_parameter_names, build_smoke_optimizer,
        make_batch)

    model = _build_cuda_prescf_detector(torch)
    try:
        model.set_epoch(0)
        model.train()
        groups, _ = _group_parameter_names(model)
        optimizer, _ = build_smoke_optimizer(model)
        batch = make_batch(2, torch.device('cuda'), 32, 64)
        ever_nonzero = {group: False for group in groups}
        required_losses = (
            '.loss_cls', '.loss_pts', '.loss_role', '.loss_ego',
            '.loss_static', '.loss_dynamic', '.loss_smooth', '.loss_leak',
            'loss_traj_')
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            losses = model.forward_train(**batch)
            for required in required_losses:
                assert any(required in name for name in losses), required
            objective = sum(
                value for value in losses.values()
                if torch.is_tensor(value) and value.requires_grad)
            assert torch.isfinite(objective)
            objective.backward()
            for group, names in groups.items():
                stats = _gradient_stats(model, names)
                assert not stats['grad_is_none'], group
                assert stats['grad_is_finite'], group
                ever_nonzero[group] |= (
                    stats['grad_abs_sum'] > 0 and
                    stats['nonzero_ratio'] > 0)
            optimizer.step()
        assert all(ever_nonzero.values()), ever_nonzero
    finally:
        del model
        torch.cuda.empty_cache()


def test_stage12_tass_parameters_do_not_change():
    """The DDP-visible staged TASS layers remain exact frozen anchors."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required to construct the detector')
    from tools.test_prescf_ddp_smoke import (
        _staged_tass_parameter_names, build_smoke_optimizer)

    model = _build_cuda_prescf_detector(torch)
    try:
        optimizer, optimizer_staged = build_smoke_optimizer(model)
        staged = _staged_tass_parameter_names(model)
        assert staged and staged == optimizer_staged
        named = dict(model.named_parameters())
        reference = {
            name: named[name].detach().clone() for name in staged}
        for epoch in (0, model.prescf_stage1_end_epoch):
            model.set_epoch(epoch)
            assert model._prescf_tass_frozen
            optimizer.zero_grad(set_to_none=True)
            # Exercise the registered gradient gate directly through
            # autograd.  Zero decay on this optimizer group must then make
            # the complete AdamW step an exact no-op.
            sum(named[name].sum() for name in staged).backward()
            assert all(named[name].grad is not None for name in staged)
            assert all(torch.count_nonzero(named[name].grad) == 0
                       for name in staged)
            optimizer.step()
            assert all(torch.equal(named[name].detach(), reference[name])
                       for name in staged)
    finally:
        del model
        torch.cuda.empty_cache()


def test_stage3_only_unfreezes_configured_tass_layers():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required to construct the detector')
    from tools.test_prescf_ddp_smoke import (
        _staged_tass_parameter_names, build_smoke_optimizer)

    model = _build_cuda_prescf_detector(torch)
    try:
        staged = _staged_tass_parameter_names(model)
        all_tass = {
            name for name, _ in model.named_parameters()
            if name.startswith('pts_bbox_head.')}
        named = dict(model.named_parameters())
        model.set_epoch(int(model.dsqe_cfg['stage3_start_epoch']))
        assert not model._prescf_tass_frozen
        assert {name for name in all_tass if named[name].requires_grad} == staged
        optimizer, optimizer_staged = build_smoke_optimizer(model)
        assert optimizer_staged == staged
        staged_group = next(
            group for group in optimizer.param_groups
            if group['group_name'] == 'tass_stage3')
        main_group = next(
            group for group in optimizer.param_groups
            if group['group_name'] == 'prescf')
        assert staged_group['lr'] == pytest.approx(main_group['lr'] * 0.1)
        assert staged_group['weight_decay'] == 0.
        optimizer.zero_grad(set_to_none=True)
        sum(named[name].sum() for name in staged).backward()
        assert all(named[name].grad is not None and
                   torch.count_nonzero(named[name].grad) > 0
                   for name in staged)
        reference = {
            name: named[name].detach().clone() for name in staged}
        optimizer.step()
        assert any(not torch.equal(named[name].detach(), reference[name])
                   for name in staged)
        assert all(not named[name].requires_grad for name in all_tass - staged)
    finally:
        del model
        torch.cuda.empty_cache()


def test_six_step_prescf_rollout_recurses_and_backpropagates():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required for the detector-level PreSCF smoke test')
    from mmcv import Config
    from mmdet3d.models import build_model

    root = Path(__file__).parents[2]
    cfg = Config.fromfile(str(
        root / 'configs/sparseworld/nuscenes-temporal/sparseworld-traj-prescf.py'))
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg')).cuda().eval()
    # A BaseLine checkpoint has no absolute PreSCF semantic head.  Loading
    # its complete point-specific MLP must warm-start every corresponding
    # layer, without averaging the final 48x17 projection.
    warm_state = {}
    for index in (0, 2, 4):
        warm_state['cls_branch.{}.weight'.format(index)] = \
            torch.randn_like(model.cls_branch[index].weight.cpu()) * 0.01
        warm_state['cls_branch.{}.bias'.format(index)] = \
            torch.randn_like(model.cls_branch[index].bias.cpu()) * 0.01
    model.load_state_dict(warm_state, strict=False)
    for index in (0, 2, 4):
        assert torch.allclose(
            getattr(model.joint_refine.semantic_head[index], 'weight'),
            warm_state['cls_branch.{}.weight'.format(index)].cuda())
        assert torch.allclose(
            getattr(model.joint_refine.semantic_head[index], 'bias'),
            warm_state['cls_branch.{}.bias'.format(index)].cuda())
    assert model.joint_refine.point_adapter_gate.item() == 1.
    for module in (model.img_backbone, model.img_neck,
                   model.plan_head, model.ego_cross_attn, model.traj_head):
        assert not any(parameter.requires_grad for parameter in module.parameters())
    decoder_layers = list(model.pts_bbox_head.transformer.decoder.decoder_layers)
    staged_ids = {
        id(parameter) for layer in decoder_layers[-2:]
        for parameter in layer.parameters()
    }
    assert staged_ids
    for parameter in model.pts_bbox_head.parameters():
        assert parameter.requires_grad == (id(parameter) in staged_ids)
    model.set_epoch(0)
    assert not model.pretrain and not model.pts_bbox_head.pretrain
    assert model.prescf_role_teacher_forcing == 1
    assert model.prescf_ego_teacher_forcing == 1
    model.pts_bbox_head.ind_stamps_all = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6], device='cuda')
    batch, queries, points, channels = 1, 7, 48, 256
    outs = dict(
        query_feat=torch.randn(batch, queries, channels, device='cuda',
                               requires_grad=True),
        all_refine_pts=[torch.rand(batch, queries, points, 3, device='cuda',
                                   requires_grad=True)],
        all_cls_scores=[torch.randn(batch, queries, points, 17, device='cuda',
                                    requires_grad=True)])
    ego_feat = torch.randn(batch, 1, channels, device='cuda',
                           requires_grad=True)
    # Poison future-GT fields: inference must never inspect them.
    poison = dict(temporal2ego=object(), temporal_adjacent2ego=object(),
                  temporal_agent_boxes=object(), temporal_agent_feats=object())
    result = model._forward_prescf(outs, ego_feat, [{}], poison)
    states = result['prescf_outputs']
    assert len(states) == 6
    for step, state in enumerate(states):
        num_queries = step + 2
        assert result['forecast_points_list'][step].shape == (
            batch, num_queries, points, 3)
        assert result['forecast_semantics_list'][step].shape == (
            batch, num_queries, points, 17)
        assert state['role_pred'].shape == (batch, num_queries, points, 1)
        assert state['query_role'].shape == (batch, num_queries, 1)
        assert state['query_motion'].shape == (batch, num_queries, 3)
        expected_semantics = model.joint_refine.predict_semantics(
            state['joint_feat'], state['points_metric'])
        assert torch.allclose(state['semantics'], expected_semantics)
    for step in range(5):
        previous, following = states[step], states[step + 1]
        count = previous['joint_feat'].shape[1]
        assert torch.allclose(
            previous['joint_feat'], following['input_feat'][:, :count])
        assert torch.allclose(
            result['forecast_points_list'][step],
            following['input_points'][:, :count])
        assert torch.allclose(
            previous['semantics'], following['input_semantics'][:, :count])

    loss = sum(state['points_metric'].square().mean()
               + state['semantics'].square().mean()
               + state['role_logits'].square().mean()
               + state['query_motion'].square().mean()
               + state['predicted_pose'].square().mean()
               for state in states)
    loss.backward()
    modules = (
        model.role_router, model.prescf_ego_pose_head,
        model.dual_evolution.motion_head, model.dual_evolution.static_head,
        model.dual_evolution.dynamic_head, model.dual_evolution.new_head,
        model.dual_interaction, model.joint_refine,
        model.joint_refine.semantic_head)
    for module in modules:
        gradients = [parameter.grad for parameter in module.parameters()
                     if parameter.requires_grad and parameter.grad is not None]
        assert gradients
        assert sum(gradient.abs().sum() for gradient in gradients) > 0
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
