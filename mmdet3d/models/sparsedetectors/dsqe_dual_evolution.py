"""Point-wise dynamic/static geometry evolution for PreSCF."""

import torch
from torch import nn

from .bbox.utils import decode_points, encode_points


class DSQEDualEvolution(nn.Module):
    def __init__(self, embed_dims, num_points, pc_range, motion_scale=4.0,
                 static_alpha=0.1, residual_scale=1.0, new_residual_scale=1.0,
                 beta=None, static_alpha_max=None,
                 dynamic_residual_scale=None, **kwargs):
        super().__init__()
        self.num_points = num_points
        self.motion_scale = motion_scale
        self.static_alpha = static_alpha
        self.residual_scale = (residual_scale if dynamic_residual_scale is None
                               else dynamic_residual_scale)
        self.new_residual_scale = new_residual_scale
        self.register_buffer('pc_range', torch.as_tensor(pc_range).float())
        self.motion_head = nn.Sequential(
            nn.Linear(embed_dims * 2 + 4, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 3))
        self.static_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_points * 3))
        self.dynamic_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_points * 3))
        self.new_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_points * 3))
        for module in (self.motion_head, self.static_head, self.dynamic_head, self.new_head):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

    @property
    def dynamic_residual_head(self):
        return self.dynamic_head

    @property
    def static_residual_head(self):
        return self.static_head

    def _residual(self, head, feat, scale):
        return head(feat).reshape(feat.shape[0], feat.shape[1], self.num_points, 3).tanh() * scale

    def forward(self, query_feat, carried_points, new_points_t0, ego_feat,
                query_role, point_role, next_to_current, next_to_t0, ego_warp,
                base_points=None):
        nc = carried_points.shape[1]
        carried_feat, new_feat = query_feat[:, :nc], query_feat[:, nc:]
        carried_prior = ego_warp.current_to_next(carried_points, next_to_current)
        new_prior = ego_warp.t0_to_next(new_points_t0, next_to_t0)
        carried_metric = decode_points(carried_prior, self.pc_range)
        new_metric = decode_points(new_prior, self.pc_range)
        if base_points is not None:
            # Compatibility adapter for archived residual experiments.  The
            # PreSCF detector never passes ``base_points``; this branch is
            # intentionally isolated and cannot become a second recursive
            # carrier.
            base_metric = decode_points(base_points, self.pc_range)
            dynamic_delta = self._residual(
                self.dynamic_residual_head, query_feat, self.residual_scale)
            dynamic_delta = dynamic_delta.clone(); dynamic_delta[..., 2] = 0
            static_delta = self._residual(
                self.static_residual_head, query_feat, self.residual_scale) * self.static_alpha
            residual_delta = point_role.to(base_metric.dtype) * dynamic_delta + \
                (1 - point_role.to(base_metric.dtype)) * static_delta
            evolved_metric = base_metric + residual_delta
            return dict(points=encode_points(evolved_metric, self.pc_range),
                        points_metric=evolved_metric,
                        base_points_metric=base_metric,
                        carried_prior_metric=carried_metric,
                        new_prior_metric=new_metric,
                        static_points_metric=base_metric + static_delta,
                        dynamic_points_metric=base_metric + dynamic_delta,
                        query_motion=carried_metric.new_zeros(
                            carried_metric.shape[:2] + (3,)),
                        static_residual=static_delta,
                        dynamic_residual=dynamic_delta,
                        new_residual=residual_delta,
                        residual_delta=residual_delta)
        if nc:
            center = carried_metric.mean(2)
            ego = ego_feat.expand(-1, nc, -1)
            motion = self.motion_head(torch.cat([
                carried_feat, ego, center, query_role[:, :nc]], -1)).tanh() * self.motion_scale
            static_delta = self._residual(self.static_head, carried_feat, self.residual_scale)
            dynamic_delta = self._residual(self.dynamic_residual_head, carried_feat, self.residual_scale)
            dynamic_delta = dynamic_delta.clone(); dynamic_delta[..., 2] = 0
            static_points = carried_metric + self.static_alpha * static_delta
            dynamic_points = carried_metric + motion.unsqueeze(2) + dynamic_delta
            gate = point_role[:, :nc].clamp(0, 1)
            carried_evolved = (1 - gate) * static_points + gate * dynamic_points
        else:
            motion = carried_metric.new_zeros(carried_metric.shape[:2] + (3,))
            static_delta = carried_metric.new_zeros(carried_metric.shape)
            dynamic_delta = carried_metric.new_zeros(carried_metric.shape)
            static_points = carried_metric
            dynamic_points = carried_metric
            carried_evolved = carried_metric
        if new_feat.shape[1]:
            new_delta = self._residual(self.new_head, new_feat, self.new_residual_scale)
            new_evolved = new_metric + new_delta
        else:
            new_delta = new_metric.new_zeros(new_metric.shape)
            new_evolved = new_metric
        points_metric = torch.cat([carried_evolved, new_evolved], 1)
        return dict(points=encode_points(points_metric, self.pc_range), points_metric=points_metric,
                    carried_prior_metric=carried_metric, new_prior_metric=new_metric,
                    static_points_metric=torch.cat([static_points, new_metric], 1),
                    dynamic_points_metric=torch.cat([dynamic_points, new_metric], 1),
                    query_motion=motion, static_residual=static_delta,
                    dynamic_residual=dynamic_delta, new_residual=new_delta)
