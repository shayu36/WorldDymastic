import torch
import torch.nn as nn


class DSQERoleRouter(nn.Module):
    """Point/query-level dynamic-static soft role routing."""

    def __init__(self,
                 embed_dims,
                 num_classes,
                 dynamic_class_ids,
                 hidden_dims=64,
                 eps=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.dynamic_class_ids = tuple(dynamic_class_ids)
        self.eps = eps

        self.query_proj = nn.Linear(embed_dims, hidden_dims)
        self.point_proj = nn.Sequential(
            nn.Linear(3, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(inplace=True))
        self.semantic_proj = nn.Sequential(
            nn.Linear(num_classes, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(inplace=True))
        self.source_embed = nn.Embedding(2, hidden_dims)

        self.context_norm = nn.LayerNorm(hidden_dims)
        self.role_head = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, 1))
        self.pool_head = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, 1))

        nn.init.zeros_(self.role_head[-1].weight)
        nn.init.zeros_(self.role_head[-1].bias)

    def semantic_dynamic_prior(self, semantic_logits):
        scores = semantic_logits.sigmoid()
        non_empty_scores = scores[..., 1:]
        normalizer = non_empty_scores.sum(dim=-1, keepdim=True).clamp_min(
            self.eps)
        dynamic_scores = scores[..., list(self.dynamic_class_ids)].sum(
            dim=-1, keepdim=True)
        return (dynamic_scores / normalizer).clamp(self.eps, 1 - self.eps)

    def forward(self,
                query_feat,
                points,
                semantic_logits,
                source_flag,
                role_prior=None,
                role_prior_valid=None,
                teacher_role=None,
                teacher_valid=None,
                teacher_forcing_ratio=0.0):
        batch_size, num_query, num_points = points.shape[:3]
        source_index = source_flag.squeeze(-1).long().clamp(0, 1)

        query_context = self.query_proj(query_feat).unsqueeze(2).expand(
            -1, -1, num_points, -1)
        point_context = self.point_proj(points)
        semantic_context = self.semantic_proj(semantic_logits)
        source_context = self.source_embed(source_index).unsqueeze(2).expand(
            -1, -1, num_points, -1)
        context = self.context_norm(
            query_context + point_context + semantic_context + source_context)

        semantic_prior = self.semantic_dynamic_prior(semantic_logits)
        routing_prior = semantic_prior
        if role_prior is not None:
            role_prior = role_prior.to(
                device=semantic_prior.device, dtype=semantic_prior.dtype)
            if role_prior_valid is None:
                role_prior_valid = torch.ones_like(
                    role_prior, dtype=torch.bool)
            role_prior_valid = role_prior_valid.to(
                device=semantic_prior.device).bool()
            routing_prior = torch.where(
                role_prior_valid,
                role_prior.clamp(self.eps, 1 - self.eps),
                semantic_prior)
        prior_logits = torch.logit(routing_prior, eps=self.eps)
        role_logits = prior_logits + self.role_head(context)
        role_pred = role_logits.sigmoid()
        pool_weights = self.pool_head(context).softmax(dim=2)

        route_role = role_pred
        if teacher_role is not None and teacher_forcing_ratio > 0:
            ratio = float(max(0.0, min(1.0, teacher_forcing_ratio)))
            if teacher_valid is None:
                teacher_valid = torch.ones_like(teacher_role, dtype=torch.bool)
            teacher_role = teacher_role.to(
                device=role_pred.device, dtype=role_pred.dtype)
            teacher_valid = teacher_valid.to(device=role_pred.device).bool()
            blended = ratio * teacher_role + (1 - ratio) * role_pred
            route_role = torch.where(teacher_valid, blended, role_pred)

        query_role = (pool_weights * route_role).sum(dim=2)
        return dict(
            semantic_prior=semantic_prior,
            routing_prior=routing_prior,
            role_logits=role_logits,
            role_pred=role_pred,
            route_role=route_role,
            query_role=query_role,
            pool_weights=pool_weights,
        )
