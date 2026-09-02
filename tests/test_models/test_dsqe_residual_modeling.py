from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn

from mmdet3d.models.sparsedetectors.dsqe_dual_evolution import (
    DSQEDualEvolution)
from mmdet3d.models.sparsedetectors.opus_head import OPUSHead
from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import (
    SparseWorld4DTraj)
from mmdet3d.models.sparsedetectors.dsqe_ego_warp import DSQEEgoWarp
from mmdet3d.models.sparsedetectors.bbox.utils import encode_points


PC_RANGE = [-10.0, -10.0, -2.0, 10.0, 10.0, 2.0]


class _ZeroResidualAdapter(nn.Module):

    def __init__(self, dim=8, points=4, classes=17):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim))
        self.point = nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, points * 3))
        self.semantic = nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, points * classes))
        for module in (self.feature, self.point, self.semantic):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

    def forward(self, base_feat, base_points, base_semantics):
        feat_delta = self.feature(base_feat)
        point_delta = self.point(base_feat).reshape(
            base_feat.shape[0], base_feat.shape[1], -1, 3)
        semantic_delta = self.semantic(base_feat).reshape(
            base_feat.shape[0], base_feat.shape[1], -1, 17)
        return (base_feat + feat_delta, base_points + point_delta,
                base_semantics + semantic_delta)


def test_baseline_residual_identity():
    torch.manual_seed(1)
    adapter = _ZeroResidualAdapter()
    feat = torch.randn(2, 3, 8)
    points = torch.randn(2, 3, 4, 3)
    semantics = torch.randn(2, 3, 4, 17)
    output = adapter(feat, points, semantics)
    torch.testing.assert_close(output[0], feat)
    torch.testing.assert_close(output[1], points)
    torch.testing.assert_close(output[2], semantics)


def test_residual_identity_has_trainable_gradient():
    adapter = _ZeroResidualAdapter()
    feat = torch.randn(1, 2, 8, requires_grad=True)
    points = torch.randn(1, 2, 4, 3)
    semantics = torch.randn(1, 2, 4, 17)
    output = adapter(feat, points, semantics)
    loss = output[0].square().mean() + output[1].square().mean()
    loss.backward()
    assert adapter.feature[-1].weight.grad is not None
    assert torch.isfinite(adapter.feature[-1].weight.grad).all()
    assert adapter.feature[-1].weight.grad.abs().sum() > 0


def test_semantics_are_absolute_per_step_not_previous_logits():
    model = SimpleNamespace(
        training=True,
        num_refines=2,
        dsqe_cfg={'semantic_residual_scale': 1.0},
        semantic_correction_head=nn.Linear(8, 34, bias=False))
    nn.init.zeros_(model.semantic_correction_head.weight)
    base = torch.randn(1, 2, 2, 17)
    residual_feat = torch.randn(1, 2, 8)
    first, _ = SparseWorld4DTraj._refine_semantics(
        model, base, residual_feat, residual_feat)
    changed_previous = base + 100.0
    second, _ = SparseWorld4DTraj._refine_semantics(
        model, base, residual_feat, residual_feat)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, changed_previous)


def test_motion_state_roles_use_speed_and_validity():
    points = torch.tensor([
        [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]
    ])
    labels = torch.tensor([4, 4, 11])
    metadata = {
        'centers': torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        'radius': torch.tensor([1.0, 1.0]),
        'role': torch.tensor([0.95, 0.1]),
        'valid': torch.tensor([True, False]),
    }
    target, valid = OPUSHead._assign_motion_state_roles(
        points, labels, metadata, [1, 8, 11, 12, 13, 14, 15, 16])
    assert target[0] > 0.9 and valid[0]
    assert not valid[1]
    assert target[2] == 0 and valid[2]


def test_mixed_query_uses_point_role_not_query_average():
    evolution = DSQEDualEvolution(
        embed_dims=8, num_points=2, pc_range=PC_RANGE)
    warp = DSQEEgoWarp(PC_RANGE)
    features = torch.randn(1, 1, 8)
    base_metric = torch.tensor([[[[0., 0., 0.], [1., 0., 0.]]]])
    base_points = encode_points(base_metric, torch.tensor(PC_RANGE))
    role = torch.tensor([[[1.0], [0.0]]]).reshape(1, 1, 2, 1)
    # Use different point roles inside one query and non-zero heads.
    with torch.no_grad():
        evolution.dynamic_residual_head[-1].bias[0] = 1.0
        evolution.static_residual_head[-1].bias[3] = -1.0
    output = evolution(
        features, base_points[:, :0], base_points, features,
        torch.tensor([[[0.5]]]), role, warp.identity(1, features.device, features.dtype),
        warp.identity(1, features.device, features.dtype), warp,
        base_points=base_points)
    assert output['residual_delta'][0, 0, 0, 0] > 0
    assert output['residual_delta'][0, 0, 1, 0] < 0


def test_dynamic_loss_soft_dead_zone_has_role_gradient():
    class _Loss(nn.Module):
        def forward(self, scores, labels, weight=None, avg_factor=None):
            return scores.sum() * 0

    head = OPUSHead.__new__(OPUSHead)
    nn.Module.__init__(head)
    head.dsqe_cfg = {'dynamic_class_ids': [4]}
    head.num_classes = 17
    head.loss_cls = _Loss()
    role_pred = torch.full((1, 1, 2, 1), 0.2, requires_grad=True)
    refine = torch.tensor([[[0., 0., 0.], [2., 0., 0.]]])
    cache = {
        'valid_gt_list': [True],
        'gt_points_list': [torch.tensor([[1., 0., 0.]])],
        'gt_labels_list': [torch.tensor([4])],
        'labels_list': [torch.tensor([0, 0])],
        'gt_paired_idx_list': [torch.tensor([0, 0])],
        'pred_paired_idx_list': [torch.tensor([0])],
        'label_weights_list': [torch.ones(2, 17)],
        'role_target': torch.ones(1, 1, 2),
        'role_valid': torch.ones(1, 1, 2, dtype=torch.bool),
        'gt_role_target_list': [torch.ones(1)],
        'gt_role_valid_list': [torch.ones(1, dtype=torch.bool)],
    }
    output = {'role_pred': role_pred}
    loss = OPUSHead._loss_dynamic(
        head, torch.zeros(1, 1, 2, 17), output, cache, refine)
    assert loss > 0
    loss.backward()
    assert role_pred.grad is not None
    assert role_pred.grad.abs().sum() > 0


def test_zero_point_residual_does_not_add_legacy_velocity():
    evolution = DSQEDualEvolution(
        embed_dims=8, num_points=2, pc_range=PC_RANGE)
    warp = DSQEEgoWarp(PC_RANGE)
    base_metric = torch.tensor([[[[1., 2., 0.], [2., 2., 0.]]]])
    base_points = encode_points(base_metric, torch.tensor(PC_RANGE))
    output = evolution(
        torch.randn(1, 1, 8), base_points[:, :0], base_points,
        torch.randn(1, 1, 8), torch.tensor([[[0.5]]]),
        torch.full((1, 1, 2, 1), 0.5), warp.identity(1, torch.device('cpu'), torch.float32),
        warp.identity(1, torch.device('cpu'), torch.float32), warp,
        base_points=base_points)
    torch.testing.assert_close(output['points_metric'], base_metric)


def test_tass_assignment_is_fixed_across_epochs():
    class _Head:
        num_stamps_all = torch.ones(8, 7, dtype=torch.long)
        ind_stamps_all = None

        @staticmethod
        def reset_mask():
            return None

    model = SimpleNamespace(
        dsqe_enabled=True,
        dsqe_cfg={'freeze_tass': True},
        pts_bbox_head=_Head(),
        num_query=2,
        num_fu_query=[1, 1, 1, 1, 1, 1])
    model._synchronize_tass_state = MethodType(
        SparseWorld4DTraj._synchronize_tass_state, model)
    model._ensure_tass_state = MethodType(
        SparseWorld4DTraj._ensure_tass_state, model)
    model._ensure_tass_state()
    first = model.pts_bbox_head.ind_stamps_all.clone()
    model._ensure_tass_state()
    torch.testing.assert_close(first, model.pts_bbox_head.ind_stamps_all)
    counts = torch.bincount(first.long(), minlength=7).tolist()
    assert counts == [2, 1, 1, 1, 1, 1, 1]


def test_old_checkpoint_missing_new_residual_head_is_compatible():
    module = _ZeroResidualAdapter()
    old_state = {key: value for key, value in module.state_dict().items()
                 if not key.startswith('feature.')}
    incompatible = module.load_state_dict(old_state, strict=False)
    assert incompatible.missing_keys
    assert all(key.startswith('feature.') for key in incompatible.missing_keys)
