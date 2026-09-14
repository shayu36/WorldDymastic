"""Shared correction and absolute-state refinement."""

import torch
from torch import nn


class DSQEJointRefine(nn.Module):
    def __init__(self, embed_dims, num_points, num_classes=17,
                 num_spatial_layers=1, num_heads=8, local_k=16, **kwargs):
        super().__init__()
        self.dynamic_proj = nn.Linear(embed_dims, embed_dims)
        self.static_proj = nn.Linear(embed_dims, embed_dims)
        self.point_proj = nn.Sequential(nn.Linear(3, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True))
        self.norm = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(nn.Linear(embed_dims, embed_dims * 2), nn.ReLU(inplace=True), nn.Linear(embed_dims * 2, embed_dims))
        if embed_dims % num_heads:
            raise ValueError('embed_dims must be divisible by num_heads')
        self.spatial_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
            for _ in range(int(num_spatial_layers))])
        self.spatial_norm = nn.ModuleList([
            nn.LayerNorm(embed_dims) for _ in range(int(num_spatial_layers))])
        self.local_k = max(int(local_k), 1)
        # Corrections are point-wise.  Zero initialization makes a loaded
        # BaseLine checkpoint an approximate identity at PreSCF start.
        self.point_correction = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 3))
        self.role_correction = nn.Linear(embed_dims, 1)
        self.semantic_head = nn.Linear(embed_dims, num_classes)
        nn.init.zeros_(self.point_correction[-1].weight)
        nn.init.zeros_(self.point_correction[-1].bias)
        nn.init.zeros_(self.role_correction.weight)
        nn.init.zeros_(self.role_correction.bias)

    def predict_semantics(self, query_feat, points_metric, point_features=None):
        """Predict absolute logits from the corrected feature/point state."""
        if point_features is None:
            point_features = query_feat.unsqueeze(2) + self.point_proj(points_metric)
        return self.semantic_head(point_features)

    def _local_spatial_attention(self, module, query, centers):
        """Exchange state across nearest Queries without dense N² attention."""
        batch_size, num_queries, channels = query.shape
        if num_queries == 0:
            return query
        k = min(max(self.local_k, 1), num_queries)
        with torch.no_grad():
            indices = torch.cdist(
                centers.detach().float(), centers.detach().float()).topk(
                    k, dim=-1, largest=False).indices
        batch = torch.arange(batch_size, device=query.device)[:, None, None]
        neighbors = query[batch, indices]
        attended = module(
            query.reshape(batch_size * num_queries, 1, channels),
            neighbors.reshape(batch_size * num_queries, k, channels),
            neighbors.reshape(batch_size * num_queries, k, channels),
            need_weights=False)[0]
        return attended.reshape(batch_size, num_queries, channels)

    def forward(self, base_feat, dynamic_feat, static_feat, points_metric):
        if base_feat.shape[1] == 0:
            empty = points_metric.new_zeros(points_metric.shape)
            return dict(query_feat=base_feat,
                        point_correction=empty,
                        role_correction=points_metric.new_zeros(
                            points_metric.shape[0], points_metric.shape[1],
                            points_metric.shape[2], 1),
                        semantic_logits=points_metric.new_zeros(
                            points_metric.shape[0], points_metric.shape[1],
                            points_metric.shape[2], self.semantic_head.out_features))
        center = points_metric.mean(2)
        joint = self.norm(base_feat + self.dynamic_proj(dynamic_feat) + self.static_proj(static_feat) + self.point_proj(center))
        joint = self.norm(joint + self.ffn(joint))
        # Shared spatial correction is Query-level: every dynamic/static view
        # has already been fused into ``joint``, and center PE lets the layer
        # exchange corrections across the common Query bank.  Expand to the
        # 48-point state only after that shared exchange.
        for attn, norm in zip(self.spatial_attn, self.spatial_norm):
            residual = joint
            joint = self._local_spatial_attention(attn, joint, center)
            joint = norm(joint + residual)
        point_state = joint.unsqueeze(2) + self.point_proj(points_metric)
        point_delta = self.point_correction(point_state).tanh()
        role_delta = self.role_correction(point_state)
        # Keep the public semantic path deterministic from the returned joint
        # Query feature and corrected coordinates; the shared spatial state
        # is used by point/role correction while this head remains compatible
        # with existing checkpoint/evaluation callers.
        semantic_logits = self.predict_semantics(joint, points_metric)
        return dict(query_feat=joint, point_correction=point_delta,
                    role_correction=role_delta,
                    semantic_logits=semantic_logits,
                    point_features=point_state)
