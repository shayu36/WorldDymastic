import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .dsqe_dual_interaction import DSQERoleSpatialAttention


class DSQEJointRefine(nn.Module):
    """Shared refinement after constrained dynamic/static evolution."""

    def __init__(self,
                 embed_dims,
                 num_points,
                 num_heads=8,
                 local_k=16,
                 dropout=0.1,
                 use_checkpoint=False):
        super().__init__()
        self.num_points = num_points
        self.use_checkpoint = use_checkpoint
        self.dynamic_proj = nn.Linear(embed_dims, embed_dims)
        self.static_proj = nn.Linear(embed_dims, embed_dims)
        self.center_encoder = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True))
        self.fusion_norm = nn.LayerNorm(embed_dims)
        self.spatial_attention = DSQERoleSpatialAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            local_k=local_k,
            dropout=dropout)
        self.attention_norm = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 2, embed_dims))
        self.ffn_norm = nn.LayerNorm(embed_dims)
        self.role_correction_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_points))

        nn.init.zeros_(self.role_correction_head[-1].weight)
        nn.init.zeros_(self.role_correction_head[-1].bias)

    def forward(self, base_feat, dynamic_feat, static_feat, points_metric):
        centers = points_metric.mean(dim=2)
        joint_feat = base_feat + self.dynamic_proj(dynamic_feat) + \
            self.static_proj(static_feat) + self.center_encoder(centers)
        joint_feat = self.fusion_norm(joint_feat)

        unit_gate = joint_feat.new_ones(*joint_feat.shape[:2], 1)
        neighbor_indices = self.spatial_attention.get_neighbors(centers)
        if self.use_checkpoint and self.training and joint_feat.requires_grad:
            def attention_forward(query, gate, centers, neighbor_indices):
                return self.spatial_attention(
                    query, query, query, gate, gate, centers,
                    neighbor_indices)

            attention_update = checkpoint(
                attention_forward, joint_feat, unit_gate, centers,
                neighbor_indices, use_reentrant=False)
        else:
            attention_update = self.spatial_attention(
                joint_feat, joint_feat, joint_feat,
                unit_gate, unit_gate, centers, neighbor_indices)
        joint_feat = self.attention_norm(joint_feat + attention_update)
        joint_feat = self.ffn_norm(joint_feat + self.ffn(joint_feat))
        role_correction = self.role_correction_head(joint_feat).unsqueeze(-1)
        return dict(
            query_feat=joint_feat,
            role_correction=role_correction,
        )
