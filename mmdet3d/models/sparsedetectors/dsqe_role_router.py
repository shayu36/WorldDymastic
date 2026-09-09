"""Differentiable dynamic/static role routing for DSQE-PreSCF."""

import torch
from torch import nn


class DSQERoleRouter(nn.Module):
    """Predict point and query level dynamic probabilities.

    The router deliberately uses soft probabilities throughout.  A Query is
    represented once; ``role_pred`` and ``query_role`` are two views of that
    same state rather than two independent Query banks.
    """

    def __init__(self, embed_dims, num_classes=17,
                 dynamic_class_ids=(2, 3, 4, 5, 6, 7, 9, 10),
                 hidden_dims=64, eps=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.dynamic_class_ids = tuple(dynamic_class_ids)
        self.eps = eps
        self.query_proj = nn.Linear(embed_dims, hidden_dims)
        self.point_proj = nn.Sequential(
            nn.Linear(3, hidden_dims), nn.LayerNorm(hidden_dims),
            nn.ReLU(inplace=True))
        self.semantic_proj = nn.Sequential(
            nn.Linear(num_classes, hidden_dims), nn.LayerNorm(hidden_dims),
            nn.ReLU(inplace=True))
        self.source_embed = nn.Embedding(2, hidden_dims)
        self.context_norm = nn.LayerNorm(hidden_dims)
        self.role_head = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, 1))
        self.pool_head = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, 1))
        nn.init.zeros_(self.role_head[-1].weight)
        nn.init.zeros_(self.role_head[-1].bias)

    def semantic_dynamic_prior(self, semantic_logits):
        scores = semantic_logits.sigmoid()
        dynamic = scores[..., list(self.dynamic_class_ids)].sum(-1, keepdim=True)
        normalizer = scores[..., 1:].sum(-1, keepdim=True).clamp_min(self.eps)
        return (dynamic / normalizer).clamp(self.eps, 1.0 - self.eps)

    def forward(self, query_feat, points, semantic_logits, source_flag,
                role_prior=None, role_prior_valid=None, teacher_role=None,
                teacher_valid=None, teacher_forcing_ratio=0.0):
        b, q, p = points.shape[:3]
        source = source_flag.squeeze(-1).long().clamp(0, 1)
        context = self.query_proj(query_feat).unsqueeze(2)
        context = context.expand(-1, -1, p, -1)
        context = context + self.point_proj(points)
        context = context + self.semantic_proj(semantic_logits)
        context = context + self.source_embed(source).unsqueeze(2)
        context = self.context_norm(context)

        semantic_prior = self.semantic_dynamic_prior(semantic_logits)
        routing_prior = semantic_prior
        if role_prior is not None:
            valid = (torch.ones_like(role_prior, dtype=torch.bool)
                     if role_prior_valid is None else role_prior_valid.bool())
            routing_prior = torch.where(
                valid, role_prior.to(semantic_prior.dtype).clamp(self.eps, 1 - self.eps),
                semantic_prior)
        role_logits = torch.logit(routing_prior, eps=self.eps) + self.role_head(context)
        role_pred = role_logits.sigmoid()
        pool_weights = self.pool_head(context).softmax(dim=2)
        route_role = role_pred
        if teacher_role is not None and teacher_forcing_ratio > 0:
            valid = (torch.ones_like(teacher_role, dtype=torch.bool)
                     if teacher_valid is None else teacher_valid.bool())
            teacher_role = teacher_role.to(role_pred.dtype)
            blended = teacher_forcing_ratio * teacher_role + (1 - teacher_forcing_ratio) * role_pred
            route_role = torch.where(valid, blended, role_pred)
        query_role = (pool_weights * route_role).sum(dim=2)
        return dict(semantic_prior=semantic_prior, routing_prior=routing_prior,
                    role_logits=role_logits, role_pred=role_pred,
                    route_role=route_role, query_role=query_role,
                    pool_weights=pool_weights)
