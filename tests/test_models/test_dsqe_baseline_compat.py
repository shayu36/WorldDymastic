from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from mmdet3d.models.sparsedetectors.bbox.utils import (
    decode_points, encode_points)
from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import (
    SparseWorld4DTraj)


PC_RANGE = torch.tensor([-10.0, -10.0, -2.0, 10.0, 10.0, 2.0])


class _EgoAttention(nn.Module):

    def forward(self, ego_point, ego_feat, query_points, query_feat):
        return ego_feat + query_feat.mean(dim=1, keepdim=True), None


def _official_trans_points(points, delta, matrix, pc_range):
    inverse = torch.linalg.inv(matrix)
    points = decode_points(points, pc_range)
    points = torch.matmul(points, matrix[..., :3, :3].transpose(-1, -2))
    points = points + matrix[..., None, :3, 3] + delta
    points = torch.matmul(points, inverse[..., :3, :3].transpose(-1, -2))
    points = points + inverse[..., None, :3, 3]
    return encode_points(points, pc_range)


def _official_baseline_reference(model, outs, ego_feat, img_metas, kwargs):
    stamps = model.pts_bbox_head.ind_stamps_all
    query_feat = outs['query_feat']
    query_pos = outs['all_refine_pts'][-1]
    query_cls = outs['all_cls_scores'][-1]
    batch_size = query_feat.shape[0]
    current_mask = stamps == 0
    current_feat = query_feat[:, current_mask]
    current_pos = query_pos[:, current_mask].detach()
    timestamp = current_pos.new_zeros(
        batch_size, model.num_query, model.num_refines, 1)

    forecast_points = []
    forecast_semantics = []
    pred_trajs = []
    forecast_masks = []
    num_future = max(
        1, min(model.curr_epoch - model.finetune_epoch + 1,
               model.num_fu_frames))

    for interval in range(num_future):
        fused_ego, _ = model.ego_cross_attn(
            ego_feat.new_full((batch_size, 1, 3), 0.5), ego_feat,
            current_pos.detach(), current_feat.detach())
        pred_trajs.append(model.traj_head(fused_ego))
        new_mask = stamps == interval + 1
        current_feat = torch.cat(
            [current_feat, query_feat[:, new_mask]], dim=1)
        current_pos = torch.cat(
            [current_pos, query_pos[:, new_mask]], dim=1).detach()
        new_timestamp = current_pos.new_full(
            (batch_size, int(new_mask.sum()), model.num_refines, 1), 0.5)
        timestamp = torch.cat([timestamp, new_timestamp], dim=1)
        position_embedding = model.position_encoder(
            torch.cat([current_pos, timestamp], dim=-1).flatten(2, 3))
        current_feat = current_feat + fused_ego + position_embedding

        regression = model.reg_branch(current_feat).reshape(
            batch_size, -1, model.num_refines, 3) * 0.5
        semantics = model.cls_branch(current_feat).reshape(
            batch_size, -1, model.num_refines, 17)
        velocity = model.vel_branch(current_feat).reshape(
            batch_size, -1, model.num_refines, 2)
        moving = ((semantics.argmax(-1) >= 2) &
                  (semantics.argmax(-1) <= 10)).unsqueeze(-1)
        regression[..., :2] += velocity * moving
        current_pos = model.refine_points(
            current_pos, regression.flatten(2, 3))
        forecast_semantics.append(semantics)
        forecast_points.append(current_pos)

        matrix = torch.as_tensor(
            np.stack([meta['ego2lidar'] for meta in img_metas]),
            dtype=current_pos.dtype)
        gt_traj = kwargs['temporal_trajs'][:, interval:interval + 1]
        offset = torch.cat([-gt_traj, torch.zeros_like(gt_traj[..., :1])],
                           dim=-1)
        mask_points = _official_trans_points(
            current_pos.flatten(1, 2), offset, matrix,
            model.pc_range).reshape_as(current_pos)
        forecast_masks.append(mask_points[..., 0] >= 0)

    return dict(
        cls_score=query_cls[:, current_mask],
        refine_pts=query_pos[:, current_mask],
        forecast_semantics_list=forecast_semantics,
        forecast_points_list=forecast_points,
        pred_trajs_list=pred_trajs,
        forecast_points_mask_list=forecast_masks)


def _build_small_baseline():
    torch.manual_seed(11)
    model = SimpleNamespace(
        num_query=2,
        num_refines=2,
        num_fu_frames=2,
        curr_epoch=1,
        finetune_epoch=0,
        pretrain=False,
        training=True,
        pc_range=PC_RANGE,
        pts_bbox_head=SimpleNamespace(
            ind_stamps_all=torch.tensor([0, 0, 1, 2])),
        ego_cross_attn=_EgoAttention(),
        position_encoder=nn.Linear(8, 4),
        reg_branch=nn.Linear(4, 6),
        cls_branch=nn.Linear(4, 34),
        vel_branch=nn.Linear(4, 4),
        traj_head=nn.Linear(4, 2))
    model.refine_points = MethodType(SparseWorld4DTraj.refine_points, model)
    model.trans_points = MethodType(SparseWorld4DTraj.trans_points, model)
    model._to_batch_tensor = SparseWorld4DTraj._to_batch_tensor
    model._baseline_foreground_mask = MethodType(
        SparseWorld4DTraj._baseline_foreground_mask, model)
    return model


def _derived_future_loss(output):
    loss = output['cls_score'].square().mean()
    for semantics, points, mask, trajectory in zip(
            output['forecast_semantics_list'],
            output['forecast_points_list'],
            output['forecast_points_mask_list'],
            output['pred_trajs_list']):
        loss = loss + semantics.square().mean() + trajectory.square().mean()
        loss = loss + points[mask].square().mean()
    return loss


def test_dsqe_off_matches_official_baseline_outputs_and_loss():
    model = _build_small_baseline()
    metric_points = torch.linspace(-8, 8, 4 * 2 * 3).reshape(1, 4, 2, 3)
    query_points = encode_points(metric_points, PC_RANGE)
    outs = {
        'query_feat': torch.randn(1, 4, 4),
        'all_refine_pts': [query_points],
        'all_cls_scores': [torch.randn(1, 4, 2, 17)],
    }
    ego_feat = torch.randn(1, 1, 4)
    img_metas = [{'ego2lidar': np.eye(4, dtype=np.float32)}]
    kwargs = {'temporal_trajs': torch.tensor([[[1.0, 0.5], [2.0, -0.5]]])}

    actual = SparseWorld4DTraj._forward_baseline_scf(
        model, outs, ego_feat, img_metas, kwargs)
    expected = _official_baseline_reference(
        model, outs, ego_feat, img_metas, kwargs)

    for key in ('cls_score', 'refine_pts'):
        torch.testing.assert_close(actual[key], expected[key])
    for key in ('forecast_semantics_list', 'forecast_points_list',
                'pred_trajs_list', 'forecast_points_mask_list'):
        assert len(actual[key]) == len(expected[key])
        for actual_value, expected_value in zip(actual[key], expected[key]):
            torch.testing.assert_close(actual_value, expected_value)
    torch.testing.assert_close(
        _derived_future_loss(actual), _derived_future_loss(expected))
