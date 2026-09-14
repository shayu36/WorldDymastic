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
        if neighbor_indices is None:
            neighbor_indices = self.get_neighbors(centers)
        bsz, num_queries, channels = query.shape
        if neighbor_indices.shape[:2] != (bsz, num_queries):
            raise ValueError('neighbor_indices must have shape [B,N_query,K]')
        batch = torch.arange(bsz, device=query.device)[:, None, None]
        gathered_key = key[batch, neighbor_indices]
        gathered_value = value[batch, neighbor_indices]
        gathered_gate = key_gate[batch, neighbor_indices].squeeze(-1)
        sender_scale = gathered_gate.unsqueeze(-1)
        gathered_key = gathered_key * sender_scale
        gathered_value = gathered_value * sender_scale
        local_k = neighbor_indices.shape[-1]
        sender_present = (gathered_gate > self.eps).any(-1)
        padding = gathered_gate <= self.eps
        safe_padding = torch.where(
            sender_present.unsqueeze(-1), padding, torch.zeros_like(padding))
        output = self.attn(
            query.reshape(bsz * num_queries, 1, channels),
            gathered_key.reshape(bsz * num_queries, local_k, channels),
            gathered_value.reshape(bsz * num_queries, local_k, channels),
            key_padding_mask=safe_padding.reshape(bsz * num_queries, local_k),
            need_weights=False)[0].reshape(bsz, num_queries, channels)
        return (output * query_gate *
                sender_present.unsqueeze(-1).to(output.dtype))


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
        self.local_k = int(local_k)

    @property
    def dynamic_from_static_gate(self):
        return self.dynamic_from_static.clamp_min(0)

    @property
    def static_from_dynamic_gate(self):
        return self.static_from_dynamic.clamp_min(0)

    def _local_attention(self, module, query, key, value,
                         query_centers, key_centers, key_gate):
        """Apply MHA only over the nearest ``local_k`` sender Queries."""
        bsz, nq, channels = query.shape
        nk = key.shape[1]
        if nq == 0:
            return query
        if nk == 0:
            return query.new_zeros(query.shape)
        k = min(self.local_k, nk)
        distances = torch.cdist(query_centers.float(), key_centers.float())
        indices = distances.topk(k, dim=-1, largest=False).indices
        batch = torch.arange(bsz, device=query.device)[:, None, None]
        gathered_key = key[batch, indices]
        gathered_value = value[batch, indices]
        gathered_gate = key_gate[batch, indices].squeeze(-1)
        # Flatten query-specific neighborhoods into an ordinary MHA batch.
        q_flat = query.reshape(bsz * nq, 1, channels)
        k_flat = gathered_key.reshape(bsz * nq, k, channels)
        v_flat = gathered_value.reshape(bsz * nq, k, channels)
        valid_sender = gathered_gate.reshape(bsz * nq, k) > 1e-6
        sender_present = valid_sender.any(-1)
        padding = ~valid_sender
        # Avoid an all-masked row (which would produce NaNs in softmax).
        padding = torch.where(sender_present.unsqueeze(-1), padding,
                              torch.zeros_like(padding))
        attended = module(q_flat, k_flat, v_flat,
                          key_padding_mask=padding,
                          need_weights=False)[0]
        attended = attended.reshape(bsz, nq, channels)
        sender_present = sender_present.reshape(bsz, nq, 1)
        return attended * sender_present.to(attended.dtype)

    def forward(self, query_feat, query_role, points_metric):
        if query_feat.shape[1] == 0:
            return dict(dynamic_feat=query_feat, static_feat=query_feat,
                        dynamic_from_static_gate=self.dynamic_from_static.clamp_min(0),
                        static_from_dynamic_gate=self.static_from_dynamic.clamp_min(0))
        dg = query_role.clamp(0, 1); sg = 1 - dg
        spatial = self.center_proj(points_metric.mean(dim=2))
        d = dg * (query_feat + self.dynamic_embed + spatial)
        s = sg * (query_feat + self.static_embed + spatial)
        centers = points_metric.mean(dim=2)
        dd = self._local_attention(self.dd, d, d, d, centers, centers, dg)
        ss = self._local_attention(self.ss, s, s, s, centers, centers, sg)
        ds = self._local_attention(self.ds, d, s, s, centers, centers, sg)
        sd = self._local_attention(self.sd, s, d, d, centers, centers, dg)
        du = dd + self.dynamic_from_static_gate * ds
        su = ss + self.static_from_dynamic_gate * sd
        d = dg * self.d_norm(d + du + self.d_ffn(d))
        s = sg * self.s_norm(s + su + self.s_ffn(s))
        return dict(dynamic_feat=d, static_feat=s,
                    dynamic_from_static_gate=self.dynamic_from_static_gate,
                    static_from_dynamic_gate=self.static_from_dynamic_gate)
