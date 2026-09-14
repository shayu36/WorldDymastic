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
        # Parameterize the weak S<-D branch as a sigmoid fraction of the
        # stronger D<-S branch, which enforces lambda_SD < lambda_DS for every
        # optimizer state (rather than relying on an unconstrained pair).
        def inv_softplus(value):
            value = max(float(value), 1e-4)
            return value + torch.log(-torch.expm1(torch.tensor(-value)))
        ds_init = max(float(dynamic_from_static_init), 1e-4)
        ratio_init = min(max(float(static_from_dynamic_init) / ds_init,
                             1e-4), 1.0 - 1e-4)
        self.lambda_ds_raw = nn.Parameter(inv_softplus(ds_init).reshape(()))
        self.lambda_sd_ratio_raw = nn.Parameter(
            torch.logit(torch.tensor(ratio_init)).reshape(()))
        self.distance_temperature_raw = nn.Parameter(torch.tensor(0.0))
        self.dd = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.ss = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.ds = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.sd = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.d_norm = nn.LayerNorm(embed_dims); self.s_norm = nn.LayerNorm(embed_dims)
        self.d_ffn = nn.Sequential(nn.Linear(embed_dims, embed_dims * 2), nn.ReLU(inplace=True), nn.Linear(embed_dims * 2, embed_dims))
        self.s_ffn = nn.Sequential(nn.Linear(embed_dims, embed_dims * 2), nn.ReLU(inplace=True), nn.Linear(embed_dims * 2, embed_dims))
        self.local_k = max(int(local_k), 1)

    @property
    def dynamic_from_static_gate(self):
        return torch.nn.functional.softplus(self.lambda_ds_raw)

    @property
    def dynamic_from_static(self):
        """Backward-compatible read-only name for the constrained gate."""
        return self.dynamic_from_static_gate

    @property
    def static_from_dynamic_gate(self):
        return self.dynamic_from_static_gate * torch.sigmoid(
            self.lambda_sd_ratio_raw)

    @property
    def static_from_dynamic(self):
        """Backward-compatible read-only name for the constrained gate."""
        return self.static_from_dynamic_gate

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Migrate checkpoints from the old unconstrained gate parameters.

        Older DSQE checkpoints stored positive gates directly as
        ``dynamic_from_static`` and ``static_from_dynamic``.  The PreSCF
        implementation stores a positive softplus scale and a sigmoid ratio
        so that ``lambda_SD < lambda_DS`` is guaranteed.  Translate the old
        values at load time and remove the legacy entries before PyTorch's
        strict-key bookkeeping runs.
        """
        old_ds = prefix + 'dynamic_from_static'
        old_sd = prefix + 'static_from_dynamic'
        new_ds = prefix + 'lambda_ds_raw'
        new_ratio = prefix + 'lambda_sd_ratio_raw'
        if new_ds not in state_dict and old_ds in state_dict:
            value = state_dict[old_ds].detach().float().clamp_min(1e-4)
            # inverse softplus(x) = x + log(-expm1(-x))
            state_dict[new_ds] = value + torch.log(-torch.expm1(-value))
        if new_ratio not in state_dict and old_sd in state_dict:
            ds_value = state_dict.get(old_ds, self.dynamic_from_static_gate)
            ds_value = ds_value.detach().float().clamp_min(1e-4)
            sd_value = state_dict[old_sd].detach().float().clamp_min(1e-4)
            ratio = (sd_value / ds_value).clamp(1e-4, 1 - 1e-4)
            state_dict[new_ratio] = torch.logit(ratio)
        state_dict.pop(old_ds, None)
        state_dict.pop(old_sd, None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def _local_attention(self, module, query, key, value,
                         query_centers, key_centers, key_gate,
                         neighbor_indices=None, selected_distances=None):
        """Apply MHA only over the nearest ``local_k`` sender Queries."""
        bsz, nq, channels = query.shape
        nk = key.shape[1]
        if nq == 0:
            return query
        if nk == 0:
            return query.new_zeros(query.shape)
        k = min(self.local_k, nk)
        if neighbor_indices is None or selected_distances is None:
            distances = torch.cdist(
                query_centers.float(), key_centers.float())
            selected_distances, neighbor_indices = distances.topk(
                k, dim=-1, largest=False)
        else:
            k = neighbor_indices.shape[-1]
        indices = neighbor_indices
        batch = torch.arange(bsz, device=query.device)[:, None, None]
        gathered_key = key[batch, indices]
        gathered_value = value[batch, indices]
        gathered_gate = key_gate[batch, indices].squeeze(-1)
        # Continuous role log-gate + metric distance bias, added to the
        # attention logits before softmax.  Merely using this value as a
        # boolean padding mask would discard both continuous signals.
        role_log_gate = torch.log(gathered_gate.clamp_min(1e-6))
        distance_bias = -selected_distances / (
            torch.nn.functional.softplus(self.distance_temperature_raw) + 1e-3)
        attention_bias = role_log_gate + distance_bias
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
        # PyTorch MHA accepts one mask per flattened sample and head.
        # Repeating the query-specific bias across heads preserves the local
        # neighbor layout while applying the same metric prior to every head.
        num_heads = int(getattr(module, 'num_heads', 1))
        attention_bias = attention_bias.reshape(bsz * nq, 1, k)
        attention_bias = attention_bias.repeat_interleave(
            num_heads, dim=0).to(dtype=q_flat.dtype)
        attention_padding = padding.unsqueeze(1).repeat_interleave(
            num_heads, dim=0)
        attention_bias = attention_bias.masked_fill(
            attention_padding, torch.finfo(attention_bias.dtype).min)
        attended = module(q_flat, k_flat, v_flat,
                          attn_mask=attention_bias,
                          need_weights=False)[0]
        attended = attended.reshape(bsz, nq, channels)
        sender_present = sender_present.reshape(bsz, nq, 1)
        return attended * sender_present.to(attended.dtype)

    def forward(self, query_feat, query_role, points_metric):
        if query_feat.shape[1] == 0:
            return dict(dynamic_feat=query_feat, static_feat=query_feat,
                        dynamic_from_static_gate=self.dynamic_from_static_gate,
                        static_from_dynamic_gate=self.static_from_dynamic_gate)
        dg = query_role.clamp(0, 1); sg = 1 - dg
        spatial = self.center_proj(points_metric.mean(dim=2))
        d = dg * (query_feat + self.dynamic_embed + spatial)
        s = sg * (query_feat + self.static_embed + spatial)
        centers = points_metric.mean(dim=2)
        # All four directions share one Query bank and one evolved center per
        # Query, so their nearest-k topology is identical.  Compute it once
        # per future step rather than materializing four O(N^2) distance
        # matrices.
        k = min(self.local_k, centers.shape[1])
        selected_distances, neighbor_indices = torch.cdist(
            centers.float(), centers.float()).topk(
                k, dim=-1, largest=False)
        local = dict(neighbor_indices=neighbor_indices,
                     selected_distances=selected_distances)
        dd = self._local_attention(
            self.dd, d, d, d, centers, centers, dg, **local)
        ss = self._local_attention(
            self.ss, s, s, s, centers, centers, sg, **local)
        ds = self._local_attention(
            self.ds, d, s, s, centers, centers, sg, **local)
        sd = self._local_attention(
            self.sd, s, d, d, centers, centers, dg, **local)
        du = dd + self.dynamic_from_static_gate * ds
        su = ss + self.static_from_dynamic_gate * sd
        d = dg * self.d_norm(d + du + self.d_ffn(d))
        s = sg * self.s_norm(s + su + self.s_ffn(s))
        return dict(dynamic_feat=d, static_feat=s,
                    dynamic_from_static_gate=self.dynamic_from_static_gate,
                    static_from_dynamic_gate=self.static_from_dynamic_gate)
