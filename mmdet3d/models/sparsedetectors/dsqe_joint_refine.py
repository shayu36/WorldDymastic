"""Shared correction and absolute-state refinement."""

import torch
from torch import nn


class DSQEJointRefine(nn.Module):
    def __init__(self, embed_dims, num_points, num_classes=17, **kwargs):
        super().__init__()
        self.dynamic_proj = nn.Linear(embed_dims, embed_dims)
        self.static_proj = nn.Linear(embed_dims, embed_dims)
        self.point_proj = nn.Sequential(nn.Linear(3, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True))
        self.norm = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(nn.Linear(embed_dims, embed_dims * 2), nn.ReLU(inplace=True), nn.Linear(embed_dims * 2, embed_dims))
        self.point_correction = nn.Sequential(nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True), nn.Linear(embed_dims, num_points * 3))
        self.role_correction = nn.Linear(embed_dims, num_points)
        self.semantic_head = nn.Linear(embed_dims, num_classes)

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
        point_delta = self.point_correction(joint).reshape(*joint.shape[:2], points_metric.shape[2], 3).tanh()
        role_delta = self.role_correction(joint).unsqueeze(-1)
        point_features = joint.unsqueeze(2) + self.point_proj(points_metric)
        semantic_logits = self.semantic_head(point_features)
        return dict(query_feat=joint, point_correction=point_delta,
                    role_correction=role_delta,
                    semantic_logits=semantic_logits)
