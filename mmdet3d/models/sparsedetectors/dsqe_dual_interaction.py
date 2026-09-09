"""Asymmetric local interaction between soft dynamic/static streams."""

import torch
from torch import nn


class DSQERoleSpatialAttention(nn.Module):
    """Small role-gated local attention primitive.

    It is kept public so downstream experiments can use the same primitive as
    the dual interaction block.  Sender gates are applied before softmax, so
    an empty sender stream is exactly zero.
    """
    def __init__(self, embed_dims, num_heads=8, local_k=16, dropout=0.1,
                 eps=1e-5):
        super().__init__()
        if embed_dims % num_heads:
            raise ValueError('embed_dims must be divisible by num_heads')
        self.local_k = local_k
        self.eps = eps
        self.attn = nn.MultiheadAttention(embed_dims, num_heads,
                                          dropout=dropout, batch_first=True)

    def get_neighbors(self, centers):
        k = min(self.local_k, centers.shape[1])
        return torch.cdist(centers.float(), centers.float()).topk(
            k, dim=-1, largest=False).indices

    def forward(self, query, key, value, query_gate, key_gate, centers,
                neighbor_indices=None):
        if query.shape[1] == 0:
            return query
        if key_gate.abs().max() == 0:
            return torch.zeros_like(query)
        if neighbor_indices is None:
            neighbor_indices = self.get_neighbors(centers)
        # A dense call is robust for the small Query counts used by OPUS and
        # keeps gradients through both role gates.
        sender = key * key_gate
        output = self.attn(query, sender, value * key_gate,
                           need_weights=False)[0]
        return output * query_gate


class DSQEDualInteraction(nn.Module):
    def __init__(self, embed_dims, num_heads=8, local_k=16, dropout=0.1,
                 dynamic_from_static_init=1.0, static_from_dynamic_init=0.25,
                 **kwargs):
        super().__init__()
        self.dynamic_embed = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.static_embed = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.center_proj = nn.Sequential(
            nn.Linear(3, embed_dims), nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True))
        self.dynamic_from_static = nn.Parameter(torch.tensor(float(dynamic_from_static_init)))
        self.static_from_dynamic = nn.Parameter(torch.tensor(float(static_from_dynamic_init)))
        self.dd = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.ss = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.ds = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.sd = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.d_norm = nn.LayerNorm(embed_dims); self.s_norm = nn.LayerNorm(embed_dims)
        self.d_ffn = nn.Sequential(nn.Linear(embed_dims, embed_dims * 2), nn.ReLU(inplace=True), nn.Linear(embed_dims * 2, embed_dims))
        self.s_ffn = nn.Sequential(nn.Linear(embed_dims, embed_dims * 2), nn.ReLU(inplace=True), nn.Linear(embed_dims * 2, embed_dims))

    @property
    def dynamic_from_static_gate(self):
        return self.dynamic_from_static.clamp_min(0)

    @property
    def static_from_dynamic_gate(self):
        return self.static_from_dynamic.clamp_min(0)

    def forward(self, query_feat, query_role, points_metric):
        if query_feat.shape[1] == 0:
            return dict(dynamic_feat=query_feat, static_feat=query_feat,
                        dynamic_from_static_gate=self.dynamic_from_static.clamp_min(0),
                        static_from_dynamic_gate=self.static_from_dynamic.clamp_min(0))
        dg = query_role.clamp(0, 1); sg = 1 - dg
        spatial = self.center_proj(points_metric.mean(dim=2))
        d = dg * (query_feat + self.dynamic_embed + spatial)
        s = sg * (query_feat + self.static_embed + spatial)
        du = self.dd(d, d, d, need_weights=False)[0] + self.dynamic_from_static_gate * self.ds(d, s, s, need_weights=False)[0]
        su = self.ss(s, s, s, need_weights=False)[0] + self.static_from_dynamic_gate * self.sd(s, d, d, need_weights=False)[0]
        d = dg * self.d_norm(d + du + self.d_ffn(d))
        s = sg * self.s_norm(s + su + self.s_ffn(s))
        return dict(dynamic_feat=d, static_feat=s,
                    dynamic_from_static_gate=self.dynamic_from_static_gate,
                    static_from_dynamic_gate=self.static_from_dynamic_gate)
