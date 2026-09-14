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
    assert '.detach()' not in dynamic_call.split('else', 1)[0]
    head = (root / 'mmdet3d/models/sparsedetectors/opus_head.py').read_text()
    assert 'gt_to_pred = self._chunked_nearest_indices' in head


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
    assert set(metrics) == {
        'role_precision', 'role_recall', 'role_f1', 'dynamic_valid_ratio'}
    assert metrics['dynamic_valid_ratio'] == 0.25


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


def test_actor_absolute_displacement_and_adjacent_ego_motion_share_future_frame():
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
    # The cache stores the displacement at each horizon, not increments:
    # t=1 -> +1m and t=2 -> +2m in E0.
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
    assert torch.allclose(metadata['yaw'][0], torch.tensor(0.2), atol=1e-6)
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
    # Explicit increment mode reproduces the legacy +1 + +2 = +3m path.
    assert torch.allclose(metadata['centers'][0, :2],
                          torch.tensor([1., 0.]), atol=1e-6)
    assert torch.allclose(metadata['yaw'][0], torch.tensor(0.3), atol=1e-6)


def test_actor_box_dimensions_are_explicitly_converted_from_nuscenes_wlh():
    torch = pytest.importorskip('torch')
    from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp
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
    # Native nuScenes (w,l,h) = (2, 4, 2): a point at x=1.9 is inside the
    # length axis after conversion, while x=2.1 is outside.
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
    assert torch.allclose(metadata['dims'][0], torch.tensor([4., 2., 2.]))


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
    for module in (model.img_backbone, model.img_neck, model.pts_bbox_head,
                   model.plan_head, model.ego_cross_attn, model.traj_head):
        assert not any(parameter.requires_grad for parameter in module.parameters())
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
