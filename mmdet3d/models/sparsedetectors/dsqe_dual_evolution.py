import math

import torch
import torch.nn as nn

from mmdet3d.models.sparsedetectors.bbox.utils import (
    decode_points, encode_points)


def _inverse_sigmoid(value):
    value = min(max(value, 1e-4), 1 - 1e-4)
    return math.log(value / (1 - value))


def _zero_last(module):
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)


class DSQEDualEvolution(nn.Module):
    """Source-aware static/dynamic point evolution."""

    def __init__(self,
                 embed_dims,
                 num_points,
                 pc_range,
                 beta=0.7,
                 static_alpha=0.1,
                 static_alpha_max=0.2,
                 motion_scale=4.0,
                 dynamic_residual_scale=1.0,
                 new_residual_scale=2.0):
        super().__init__()
        self.num_points = num_points
        self.beta = beta
        self.static_alpha_max = static_alpha_max
        self.motion_scale = motion_scale
        self.dynamic_residual_scale = dynamic_residual_scale
        self.new_residual_scale = new_residual_scale
        self.register_buffer('pc_range', torch.as_tensor(pc_range).float())

        static_ratio = static_alpha / static_alpha_max
        self.static_alpha_logit = nn.Parameter(torch.tensor(
            _inverse_sigmoid(static_ratio), dtype=torch.float32))

        self.motion_head = nn.Sequential(
            nn.Linear(embed_dims * 2 + 4, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 3))
        self.static_residual_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_points * 3))
        self.dynamic_residual_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_points * 3))
        self.new_update_head = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_points * 3))

        _zero_last(self.motion_head)
        _zero_last(self.static_residual_head)
        _zero_last(self.dynamic_residual_head)
        _zero_last(self.new_update_head)

    @property
    def static_alpha(self):
        return self.static_alpha_max * self.static_alpha_logit.sigmoid()

    def _point_residual(self, head, query_feat, scale):
        residual = head(query_feat).reshape(
            query_feat.shape[0], query_feat.shape[1], self.num_points, 3)
        return residual.tanh() * scale

    def forward(self,
                query_feat,
                carried_points,
                new_points_t0,
                ego_feat,
                query_role,
                point_role,
                next_to_current,
                next_to_t0,
                ego_warp,
                legacy_dynamic_xy=None):
        num_carried = carried_points.shape[1]
        carried_feat = query_feat[:, :num_carried]
        new_feat = query_feat[:, num_carried:]

        carried_prior = ego_warp.current_to_next(
            carried_points, next_to_current)
        new_prior = ego_warp.t0_to_next(new_points_t0, next_to_t0)
        carried_prior_metric = decode_points(carried_prior, self.pc_range)
        new_prior_metric = decode_points(new_prior, self.pc_range)

        if num_carried > 0:
            center = carried_prior_metric.mean(dim=2)
            ego_context = ego_feat.expand(-1, num_carried, -1)
            motion_input = torch.cat([
                carried_feat, ego_context, center,
                query_role[:, :num_carried]
            ], dim=-1)
            query_motion = self.motion_head(motion_input).tanh()
            query_motion = query_motion * self.motion_scale

            static_residual = self._point_residual(
                self.static_residual_head, carried_feat,
                self.dynamic_residual_scale)
            dynamic_residual = self._point_residual(
                self.dynamic_residual_head, carried_feat,
                self.dynamic_residual_scale)
            if legacy_dynamic_xy is not None:
                legacy_xy = legacy_dynamic_xy[:, :num_carried].tanh()
                legacy_xy = legacy_xy * self.dynamic_residual_scale
                dynamic_residual = dynamic_residual.clone()
                dynamic_residual[..., :2] += legacy_xy

            static_points = carried_prior_metric + \
                self.static_alpha * static_residual
            dynamic_points = carried_prior_metric + \
                query_motion.unsqueeze(2) + dynamic_residual
            point_gate = self.beta * query_role[:, :num_carried].unsqueeze(2)
            point_gate = point_gate + (1 - self.beta) * \
                point_role[:, :num_carried]
            carried_evolved = (1 - point_gate) * static_points + \
                point_gate * dynamic_points
        else:
            empty_shape = (*carried_prior_metric.shape[:2], 3)
            query_motion = carried_prior_metric.new_zeros(empty_shape)
            static_residual = carried_prior_metric.new_zeros(
                carried_prior_metric.shape)
            dynamic_residual = carried_prior_metric.new_zeros(
                carried_prior_metric.shape)
            static_points = carried_prior_metric
            dynamic_points = carried_prior_metric
            carried_evolved = carried_prior_metric

        if new_feat.shape[1] > 0:
            new_ego = ego_feat.expand(-1, new_feat.shape[1], -1)
            new_context = torch.cat([new_feat, new_ego], dim=-1)
            new_residual = self._point_residual(
                self.new_update_head, new_context, self.new_residual_scale)
            new_evolved = new_prior_metric + new_residual
        else:
            new_residual = new_prior_metric.new_zeros(new_prior_metric.shape)
            new_evolved = new_prior_metric

        evolved_metric = torch.cat([carried_evolved, new_evolved], dim=1)
        evolved_points = encode_points(evolved_metric, self.pc_range)
        return dict(
            points=evolved_points,
            points_metric=evolved_metric,
            carried_prior=carried_prior,
            carried_prior_metric=carried_prior_metric,
            new_prior=new_prior,
            new_prior_metric=new_prior_metric,
            static_points_metric=static_points,
            dynamic_points_metric=dynamic_points,
            query_motion=query_motion,
            static_residual=static_residual,
            dynamic_residual=dynamic_residual,
            new_residual=new_residual,
        )
