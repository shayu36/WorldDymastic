import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _gate_logit(value, maximum=2.0):
    ratio = min(max(value / maximum, 1e-4), 1 - 1e-4)
    return math.log(ratio / (1 - ratio))


class DSQERoleSpatialAttention(nn.Module):
    """Memory-bounded local attention with spatial and role priors."""

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 local_k=16,
                 dropout=0.1,
                 eps=1e-5):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError('embed_dims must be divisible by num_heads')
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.local_k = local_k
        self.eps = eps

        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.tau_head = nn.Linear(embed_dims, num_heads)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.tau_head.weight)
        nn.init.constant_(self.tau_head.bias, -3.0)

    @staticmethod
    def _gather(tensor, indices):
        batch_index = torch.arange(
            tensor.shape[0], device=tensor.device)[:, None, None]
        return tensor[batch_index, indices]

    def get_neighbors(self, centers):
        num_query = centers.shape[1]
        num_neighbors = min(self.local_k, num_query)
        with torch.no_grad():
            distances = torch.cdist(centers.float(), centers.float())
            return distances.topk(
                num_neighbors, dim=-1, largest=False, sorted=False).indices

    def forward(self,
                query,
                key,
                value,
                query_gate,
                key_gate,
                centers,
                neighbor_indices=None):
        batch_size, num_query = query.shape[:2]
        if num_query == 0:
            return query
        if neighbor_indices is None:
            neighbor_indices = self.get_neighbors(centers)

        query_proj = self.q_proj(query).reshape(
            batch_size, num_query, self.num_heads, self.head_dims)
        key_proj = self.k_proj(key).reshape(
            batch_size, num_query, self.num_heads, self.head_dims)
        value_proj = self.v_proj(value).reshape(
            batch_size, num_query, self.num_heads, self.head_dims)
        neighbor_key = self._gather(key_proj, neighbor_indices)
        neighbor_value = self._gather(value_proj, neighbor_indices)
        neighbor_gate = self._gather(key_gate, neighbor_indices)
        neighbor_value = neighbor_value * neighbor_gate.unsqueeze(-1)

        logits = torch.einsum(
            'bnhd,bnlhd->bnhl', query_proj, neighbor_key)
        logits = logits / math.sqrt(self.head_dims)

        neighbor_centers = self._gather(centers, neighbor_indices)
        distance_sq = (
            centers.unsqueeze(2) - neighbor_centers).square().sum(dim=-1)
        tau = F.softplus(self.tau_head(query)).unsqueeze(-1)
        logits = logits - tau * distance_sq.unsqueeze(2)

        logits = logits + torch.log(
            neighbor_gate.clamp_min(self.eps)).squeeze(-1).unsqueeze(2)

        attention = self.dropout(logits.softmax(dim=-1))
        output = torch.einsum(
            'bnhl,bnlhd->bnhd', attention, neighbor_value)
        output = output.reshape(batch_size, num_query, self.embed_dims)
        output = self.out_proj(output)
        return output * query_gate


class DSQEDualInteraction(nn.Module):
    """Asymmetric D<-D, S<-S, D<-S and S<-D interactions."""

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 local_k=16,
                 dropout=0.1,
                 dynamic_from_static_init=1.0,
                 static_from_dynamic_init=0.25,
                 use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.dynamic_embed = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.static_embed = nn.Parameter(torch.zeros(1, 1, embed_dims))

        attention_args = dict(
            embed_dims=embed_dims,
            num_heads=num_heads,
            local_k=local_k,
            dropout=dropout)
        self.attn_dd = DSQERoleSpatialAttention(**attention_args)
        self.attn_ss = DSQERoleSpatialAttention(**attention_args)
        self.attn_ds = DSQERoleSpatialAttention(**attention_args)
        self.attn_sd = DSQERoleSpatialAttention(**attention_args)

        self.dynamic_from_static_logit = nn.Parameter(torch.tensor(
            _gate_logit(dynamic_from_static_init), dtype=torch.float32))
        self.static_from_dynamic_logit = nn.Parameter(torch.tensor(
            _gate_logit(static_from_dynamic_init), dtype=torch.float32))

        self.dynamic_norm1 = nn.LayerNorm(embed_dims)
        self.dynamic_norm2 = nn.LayerNorm(embed_dims)
        self.static_norm1 = nn.LayerNorm(embed_dims)
        self.static_norm2 = nn.LayerNorm(embed_dims)
        self.dynamic_ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 2, embed_dims))
        self.static_ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 2, embed_dims))

    def _attention(self, module, query, key, value, query_gate, key_gate,
                   centers, neighbor_indices):
        if self.use_checkpoint and self.training and query.requires_grad:
            def attention_forward(query, key, value, query_gate, key_gate,
                                  centers, neighbor_indices):
                return module(
                    query, key, value, query_gate, key_gate, centers,
                    neighbor_indices)

            return checkpoint(
                attention_forward, query, key, value, query_gate, key_gate,
                centers, neighbor_indices, use_reentrant=False)
        return module(
            query, key, value, query_gate, key_gate, centers,
            neighbor_indices)

    @property
    def dynamic_from_static_gate(self):
        return 2 * self.dynamic_from_static_logit.sigmoid()

    @property
    def static_from_dynamic_gate(self):
        return 2 * self.static_from_dynamic_logit.sigmoid()

    def forward(self, query_feat, query_role, points_metric):
        dynamic_gate = query_role.clamp(0, 1)
        static_gate = 1 - dynamic_gate
        dynamic_feat = dynamic_gate * (query_feat + self.dynamic_embed)
        static_feat = static_gate * (query_feat + self.static_embed)
        centers = points_metric.mean(dim=2)
        neighbor_indices = self.attn_dd.get_neighbors(centers)

        dynamic_update = self._attention(
            self.attn_dd, dynamic_feat, dynamic_feat, dynamic_feat,
            dynamic_gate, dynamic_gate, centers, neighbor_indices)
        dynamic_update = dynamic_update + self.dynamic_from_static_gate * \
            self._attention(
                self.attn_ds, dynamic_feat, static_feat, static_feat,
                dynamic_gate, static_gate, centers, neighbor_indices)

        static_update = self._attention(
            self.attn_ss, static_feat, static_feat, static_feat,
            static_gate, static_gate, centers, neighbor_indices)
        static_update = static_update + self.static_from_dynamic_gate * \
            self._attention(
                self.attn_sd, static_feat, dynamic_feat, dynamic_feat,
                static_gate, dynamic_gate, centers, neighbor_indices)

        dynamic_feat = self.dynamic_norm1(dynamic_feat + dynamic_update)
        dynamic_feat = dynamic_gate * self.dynamic_norm2(
            dynamic_feat + self.dynamic_ffn(dynamic_feat))
        static_feat = self.static_norm1(static_feat + static_update)
        static_feat = static_gate * self.static_norm2(
            static_feat + self.static_ffn(static_feat))
        return dict(
            dynamic_feat=dynamic_feat,
            static_feat=static_feat,
            dynamic_from_static_gate=self.dynamic_from_static_gate,
            static_from_dynamic_gate=self.static_from_dynamic_gate,
        )
