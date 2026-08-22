# Copyright (c) Phigent Robotics. All rights reserved.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob
from mmdet.models import DETECTORS

from mmdet3d.models import builder
from mmdet3d.models.detectors.loss import l2_loss
from mmdet3d.models.sparsedetectors.bbox.utils import (
    decode_points, encode_points, get_matched_inds)

from .dsqe_dual_evolution import DSQEDualEvolution
from .dsqe_dual_interaction import DSQEDualInteraction
from .dsqe_ego_warp import DSQEEgoWarp
from .dsqe_joint_refine import DSQEJointRefine
from .dsqe_role_router import DSQERoleRouter
from .opus import OPUS
from .opus_transformer import OPUSCrossAttention


nusc_class_frequencies = np.array([
    1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808,
    2457947, 497017, 2731022, 7224789, 214411435, 5565043, 63191967,
    76098082, 128860031, 141625221, 2307405309
])


def _linear_schedule(initial_value, epoch, start_epoch, end_epoch):
    if initial_value <= 0 or epoch >= end_epoch:
        return 0.0
    if epoch <= start_epoch:
        return float(initial_value)
    progress = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
    return float(initial_value) * (1 - progress)


@DETECTORS.register_module()
class SparseWorld4DTraj(OPUS):
    def __init__(self,
                 out_dim=32,
                 dataset_type='Nuscenes',
                 num_classes=18,
                 test_threshold=8.5,
                 drop_out=0.1,
                 use_3d_loss=True,
                 if_pretrain=False,
                 if_render=True,
                 if_post_finetune=False,
                 finetune_epoch=0,
                 num_out_query=600,
                 empty_idx=17,
                 use_focal_loss=True,
                 balance_cls_weight=True,
                 final_softplus=True,
                 dsqe_cfg=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.dataset_type = dataset_type
        self.out_dim = out_dim
        self.use_3d_loss = use_3d_loss
        self.test_threshold = test_threshold
        self.num_refines = self.pts_bbox_head.transformer.num_refines[-1]
        self.balance_cls_weight = balance_cls_weight
        self.final_softplus = final_softplus
        self.if_render = if_render
        self.if_post_finetune = if_post_finetune
        self.empty_idx = empty_idx
        self.finetune_epoch = finetune_epoch
        self.curr_epoch = 0

        if self.balance_cls_weight:
            class_weights = torch.from_numpy(
                1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
            self.semantic_loss = nn.CrossEntropyLoss(
                weight=class_weights, reduction='mean')
        else:
            self.semantic_loss = nn.CrossEntropyLoss(reduction='mean')
        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss = builder.build_loss(dict(type='CustomFocalLoss'))

        self.velocity_dim = 3
        self.past_frame = 5
        self.pc_range = self.pts_bbox_head.pc_range
        self.plan_head = nn.Sequential(
            nn.Linear(self.velocity_dim * (self.past_frame + 2), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.out_dim))
        self.ego_cross_attn = OPUSCrossAttention(
            self.out_dim, 8, drop_out, self.pts_bbox_head.pc_range)
        self.position_encoder = nn.Sequential(
            nn.Linear(4 * self.num_refines, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True))
        self.reg_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 3))
        self.vel_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 2))
        self.cls_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 17))
        self.points_scale_branch = nn.Sequential(
            nn.Linear(self.out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3))
        self.traj_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, 2))
        self.l2_loss = l2_loss()

        self.dsqe_enabled = dsqe_cfg is not None and \
            dsqe_cfg.get('enabled', True)
        self.dsqe_cfg = dict(dsqe_cfg or {})
        self.role_teacher_forcing_ratio = 0.0
        self.ego_teacher_forcing_ratio = 0.0
        if self.dsqe_enabled:
            # DSQE predicts future semantics with semantic_correction_head.
            # The legacy branch is only used by the DSQE-off baseline path;
            # leaving it trainable makes DDP require gradients that this
            # forward graph can never produce.
            self.cls_branch.requires_grad_(False)
            self._init_dsqe(drop_out)

    def _init_dsqe(self, drop_out):
        cfg = self.dsqe_cfg
        dynamic_ids = cfg.get(
            'dynamic_class_ids', [2, 3, 4, 5, 6, 7, 9, 10])
        static_ids = cfg.get(
            'static_class_ids', [1, 8, 11, 12, 13, 14, 15, 16])
        cfg['dynamic_class_ids'] = dynamic_ids
        cfg['static_class_ids'] = static_ids
        cfg.setdefault('frame_mode', 'future_ego')
        cfg.setdefault('role_teacher_forcing', 1.0)
        cfg.setdefault('ego_teacher_forcing', 1.0)
        cfg.setdefault('teacher_forcing_start_epoch', self.finetune_epoch)
        cfg.setdefault('teacher_forcing_end_epoch', self.finetune_epoch + 12)
        cfg.setdefault('point_correction_scale', 0.5)
        cfg.setdefault('semantic_residual_scale', 0.5)
        cfg.setdefault('role_correction_scale', 0.25)
        cfg.setdefault('correction_static_scale', 0.2)
        cfg.setdefault('stream_dropout', 0.1)

        num_heads = cfg.get('num_heads', 8)
        local_k = cfg.get('local_k', 16)
        dropout = cfg.get('dropout', drop_out)
        self.role_router = DSQERoleRouter(
            embed_dims=self.out_dim,
            num_classes=17,
            dynamic_class_ids=dynamic_ids,
            hidden_dims=cfg.get('role_hidden_dims', 64))
        self.ego_warp = DSQEEgoWarp(
            self.pc_range, frame_mode=cfg['frame_mode'])
        self.dual_evolution = DSQEDualEvolution(
            embed_dims=self.out_dim,
            num_points=self.num_refines,
            pc_range=self.pc_range,
            beta=cfg.get('beta', 0.7),
            static_alpha=cfg.get('static_alpha', 0.1),
            static_alpha_max=cfg.get('static_alpha_max', 0.2),
            motion_scale=cfg.get('motion_scale', 4.0),
            dynamic_residual_scale=cfg.get(
                'dynamic_residual_scale', 1.0),
            new_residual_scale=cfg.get('new_residual_scale', 2.0))
        self.dual_interaction = DSQEDualInteraction(
            embed_dims=self.out_dim,
            num_heads=num_heads,
            local_k=local_k,
            dropout=dropout,
            dynamic_from_static_init=cfg.get(
                'dynamic_from_static_init', 1.0),
            static_from_dynamic_init=cfg.get(
                'static_from_dynamic_init', 0.25),
            use_checkpoint=cfg.get('use_checkpoint', False))
        self.joint_refine = DSQEJointRefine(
            embed_dims=self.out_dim,
            num_points=self.num_refines,
            num_heads=num_heads,
            local_k=local_k,
            dropout=dropout,
            use_checkpoint=cfg.get('use_checkpoint', False))

        self.semantic_correction_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 17))

        self.ego_cross_attn_static = OPUSCrossAttention(
            self.out_dim, num_heads, dropout, self.pc_range)
        self.ego_dynamic_proj = nn.Linear(self.out_dim, self.out_dim)
        self.ego_static_proj = nn.Linear(self.out_dim, self.out_dim)
        self.ego_fusion_norm = nn.LayerNorm(self.out_dim)
        self.source_embedding = nn.Embedding(2, self.out_dim)
        self.activation_embedding = nn.Embedding(
            self.num_fu_frames + 1, self.out_dim)
        self.yaw_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, 2))
        self.pts_bbox_head.dsqe_cfg = cfg

        with torch.no_grad():
            identity = torch.eye(self.out_dim)
            self.ego_dynamic_proj.weight.copy_(identity * 0.5)
            self.ego_static_proj.weight.copy_(identity * 0.5)
            self.ego_dynamic_proj.bias.zero_()
            self.ego_static_proj.bias.zero_()
            self.yaw_head[-1].weight.zero_()
            self.yaw_head[-1].bias.copy_(torch.tensor([0.0, 1.0]))
            self.semantic_correction_head[-1].weight.zero_()
            self.semantic_correction_head[-1].bias.zero_()

    def init_weights(self):
        self.pts_bbox_head.init_weights()
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)
        if self.dsqe_enabled:
            self.ego_cross_attn.init_weights()
            self.ego_cross_attn_static.init_weights()

    def set_epoch(self, epoch):
        self.curr_epoch = epoch
        if epoch < self.finetune_epoch:
            self.pretrain = True
            self.pts_bbox_head.pretrain = True
            if getattr(self.pts_bbox_head, 'num_stamps_all', None) is not None:
                self.pts_bbox_head.num_stamps_all[:] = 1
        else:
            self.pretrain = False
            self.pts_bbox_head.pretrain = False
            num_stamps = self.pts_bbox_head.num_stamps_all / torch.sum(
                self.pts_bbox_head.num_stamps_all, dim=-1, keepdim=True)
            self.pts_bbox_head.ind_stamps_all = get_matched_inds(
                num_stamps, [self.num_query] + self.num_fu_query)
            self.pts_bbox_head.reset_mask()

        if self.dsqe_enabled:
            start = self.dsqe_cfg['teacher_forcing_start_epoch']
            end = self.dsqe_cfg['teacher_forcing_end_epoch']
            self.role_teacher_forcing_ratio = _linear_schedule(
                self.dsqe_cfg['role_teacher_forcing'], epoch, start, end)
            self.ego_teacher_forcing_ratio = _linear_schedule(
                self.dsqe_cfg['ego_teacher_forcing'], epoch, start, end)

    def refine_points(self, points_proposal, points_delta):
        batch_size, num_query = points_delta.shape[:2]
        points_delta = points_delta.reshape(
            batch_size, num_query, self.num_refines, 3)
        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        return encode_points(points_proposal + points_delta, self.pc_range)

    def trans_points(self, points_proposal, points_delta, trans_matrix):
        """Apply the official SCF trajectory mask transform."""
        inverse = torch.linalg.inv(trans_matrix)
        points = decode_points(points_proposal, self.pc_range)
        points = torch.matmul(
            points, trans_matrix[..., :3, :3].transpose(-1, -2))
        points = points + trans_matrix[..., None, :3, 3] + points_delta
        points = torch.matmul(
            points, inverse[..., :3, :3].transpose(-1, -2))
        points = points + inverse[..., None, :3, 3]
        return encode_points(points, self.pc_range)

    def _baseline_foreground_mask(self, points, interval, img_metas, kwargs):
        """Reproduce the official DSQE-off future range mask exactly."""
        ego_to_lidar = self._to_batch_tensor(
            [meta['ego2lidar'] for meta in img_metas],
            points.device, points.dtype)
        gt_traj = kwargs['temporal_trajs'][:, interval:interval + 1, :]
        gt_traj = gt_traj.to(device=points.device, dtype=points.dtype)
        trajectory_offset = torch.cat([
            -gt_traj, torch.zeros_like(gt_traj[..., :1])
        ], dim=-1)
        transformed = self.trans_points(
            points.flatten(1, 2), trajectory_offset,
            ego_to_lidar).reshape_as(points)
        return transformed[..., 0] >= 0

    def loss_traj(self, pred_traj, gt_traj, ego_interval):
        return {
            'loss_traj_{}s'.format(ego_interval):
            self.l2_loss(pred_traj, gt_traj)
        }

    @staticmethod
    def _dict_value(data, key):
        if key in data:
            return data[key]
        return data[str(key)]

    @staticmethod
    def _to_batch_tensor(value, device, dtype):
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device, dtype=dtype)
        elif isinstance(value, (list, tuple)) and value and \
                isinstance(value[0], torch.Tensor):
            tensor = torch.stack(value).to(device=device, dtype=dtype)
        else:
            tensor = torch.as_tensor(
                np.asarray(value), device=device, dtype=dtype)
        while tensor.ndim > 3 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _get_ego_feat(self, kwargs, batch_size, device, dtype):
        ego_states_all = kwargs['temporal_ego_states']
        ego_states = self._dict_value(ego_states_all, 0) \
            if isinstance(ego_states_all, dict) else ego_states_all[0]
        ego_states = self._to_batch_tensor(ego_states, device, dtype)
        if ego_states.ndim == 2:
            ego_states = ego_states.unsqueeze(1)
        ego_states = ego_states.reshape(batch_size, 1, -1)
        return self.plan_head(ego_states)

    def _get_gt_ego_poses(self, img_metas, kwargs, device, dtype):
        future_poses = kwargs.get('temporal_ego2global')
        if future_poses is None or not all(
                'ego2global' in meta for meta in img_metas):
            return None, None

        pose_dtype = torch.float64
        current = self._to_batch_tensor(
            [meta['ego2global'] for meta in img_metas],
            device, pose_dtype)
        sequence = [current]
        for interval in range(self.num_fu_frames):
            value = self._dict_value(future_poses, interval)
            sequence.append(self._to_batch_tensor(
                value, device, pose_dtype))
        sequence = torch.stack(sequence, dim=1)
        relative_matrices, relative_poses = \
            self.ego_warp.build_relative_targets(sequence)
        return (
            relative_matrices.to(dtype=dtype),
            relative_poses.to(dtype=dtype))

    def _get_lidar_to_ego(self, img_metas, device, dtype):
        if not all('ego2lidar' in meta for meta in img_metas):
            raise KeyError(
                'DSQE trajectory conversion requires img_metas["ego2lidar"]')
        ego_to_lidar = self._to_batch_tensor(
            [meta['ego2lidar'] for meta in img_metas],
            device, torch.float64)
        return self.ego_warp.inverse(ego_to_lidar).to(dtype=dtype)

    def _get_future_voxels(self, kwargs):
        temporal_semantics = kwargs.get('temporal_semantics')
        if temporal_semantics is None:
            return None
        output = []
        for interval in range(1, self.num_fu_frames + 1):
            value = self._dict_value(temporal_semantics, interval)
            output.append(value['voxel_semantics'])
        return output

    def _apply_stream_dropout(self, dynamic_feat, static_feat):
        probability = self.dsqe_cfg.get('stream_dropout', 0.0)
        if not self.training or probability <= 0:
            return dynamic_feat, static_feat
        sample = torch.rand(
            dynamic_feat.shape[0], 1, 1, device=dynamic_feat.device)
        dynamic_keep = (sample >= probability / 2).to(dynamic_feat.dtype)
        static_drop = (sample >= probability / 2) & (sample < probability)
        static_keep = (~static_drop).to(static_feat.dtype)
        return dynamic_feat * dynamic_keep, static_feat * static_keep

    def _predict_pose(self, ego_feat, query_feat, query_points, query_role,
                      geometry_cumulative, lidar_to_ego):
        dynamic_feat = query_role * (
            query_feat + self.dual_interaction.dynamic_embed)
        static_feat = (1 - query_role) * (
            query_feat + self.dual_interaction.static_embed)
        dynamic_feat, static_feat = self._apply_stream_dropout(
            dynamic_feat, static_feat)
        ego_point = query_points.new_full(
            (query_points.shape[0], 1, 3), 0.5)
        ego_dynamic, _ = self.ego_cross_attn(
            ego_point, ego_feat, query_points, dynamic_feat)
        ego_static, _ = self.ego_cross_attn_static(
            ego_point, ego_feat, query_points, static_feat)
        ego_next = self.ego_fusion_norm(
            ego_feat + self.ego_dynamic_proj(ego_dynamic - ego_feat) +
            self.ego_static_proj(ego_static - ego_feat))

        trajectory_delta = self.traj_head(ego_next)
        relative_yaw = F.normalize(
            self.yaw_head(ego_next), dim=-1, eps=1e-6).squeeze(1)
        lidar_delta, relative_pose, relative_matrix = \
            self.ego_warp.trajectory_to_ego_relative(
                trajectory_delta.squeeze(1),
                geometry_cumulative,
                lidar_to_ego,
                relative_yaw)
        return ego_next, trajectory_delta, relative_pose, \
            relative_matrix, lidar_delta

    def _select_geometry_pose(self, predicted, target):
        if target is None or not self.training or \
                self.ego_teacher_forcing_ratio <= 0:
            return predicted
        batch_size = predicted.shape[0]
        use_target = torch.rand(
            batch_size, device=predicted.device) < \
            self.ego_teacher_forcing_ratio
        return torch.where(use_target[:, None, None], target, predicted)

    def _build_role_teacher(self, previous_match, num_new, reference):
        if previous_match is None or self.role_teacher_forcing_ratio <= 0:
            return None, None
        teacher = previous_match['role_target'].unsqueeze(-1)
        valid = previous_match['role_valid'].unsqueeze(-1)
        if num_new > 0:
            new_shape = (
                teacher.shape[0], num_new, self.num_refines, 1)
            teacher = torch.cat([
                teacher, reference.new_zeros(new_shape)
            ], dim=1)
            valid = torch.cat([
                valid, torch.zeros(new_shape, device=valid.device,
                                   dtype=torch.bool)
            ], dim=1)
        return teacher, valid

    def _refine_semantics(self, base_semantics, next_feat):
        batch_size = next_feat.shape[0]
        correction = self.semantic_correction_head(next_feat).reshape(
            batch_size, -1, self.num_refines, 17)
        semantics = base_semantics + self.dsqe_cfg[
            'semantic_residual_scale'] * correction
        return semantics, correction

    def _forward_dsqe_scf(self, outs, ego_feat, img_metas, kwargs):
        ind_stamps_all = self.pts_bbox_head.ind_stamps_all
        query_feat_all = outs['query_feat']
        query_pos_all = outs['all_refine_pts'][-1]
        query_cls_all = outs['all_cls_scores'][-1]
        batch_size = query_feat_all.shape[0]

        current_mask = ind_stamps_all == 0
        state_feat = query_feat_all[:, current_mask]
        state_points = query_pos_all[:, current_mask].detach()
        state_semantics = query_cls_all[:, current_mask]
        state_role = None
        activation_step = torch.zeros(
            batch_size, state_feat.shape[1], device=state_feat.device,
            dtype=torch.long)

        gt_relative_matrices, gt_relative_poses = self._get_gt_ego_poses(
            img_metas, kwargs, state_feat.device, state_feat.dtype)
        lidar_to_ego = self._get_lidar_to_ego(
            img_metas, state_feat.device, state_feat.dtype)
        future_voxels = self._get_future_voxels(kwargs) \
            if self.training else None
        geometry_cumulative = self.ego_warp.identity(
            batch_size, state_feat.device, state_feat.dtype)

        forecast_points_list = []
        forecast_semantics_list = []
        pred_trajs_list = []
        forecast_points_mask_list = []
        dsqe_outputs = []
        match_cache_list = []
        previous_match = None

        if self.training:
            num_fu_frames = max(
                1, min(self.curr_epoch - self.finetune_epoch + 1,
                       self.num_fu_frames))
        else:
            num_fu_frames = self.num_fu_frames

        for interval in range(num_fu_frames):
            new_mask = ind_stamps_all == interval + 1
            new_feat = query_feat_all[:, new_mask]
            new_points_t0 = query_pos_all[:, new_mask].detach()
            new_semantics = query_cls_all[:, new_mask]
            num_carried = state_feat.shape[1]
            num_new = new_feat.shape[1]

            new_points_current = self.ego_warp.t0_to_next(
                new_points_t0, geometry_cumulative)
            route_points = torch.cat(
                [state_points, new_points_current], dim=1)
            base_feat = torch.cat([state_feat, new_feat], dim=1)
            base_semantics = torch.cat(
                [state_semantics, new_semantics], dim=1)
            source_flag = torch.cat([
                route_points.new_ones(batch_size, num_carried, 1),
                route_points.new_zeros(batch_size, num_new, 1)
            ], dim=1)
            new_activation = torch.full(
                (batch_size, num_new), interval + 1,
                device=activation_step.device, dtype=torch.long)
            active_step = torch.cat(
                [activation_step, new_activation], dim=1)
            timestamp = active_step.to(route_points.dtype) / \
                max(self.num_fu_frames, 1)
            timestamp = timestamp[:, :, None, None].expand(
                -1, -1, self.num_refines, -1)
            position_embedding = self.position_encoder(
                torch.cat([route_points, timestamp], dim=-1).flatten(2, 3))
            conditioned_feat = base_feat + position_embedding + \
                self.source_embedding(source_flag.squeeze(-1).long()) + \
                self.activation_embedding(active_step)

            role_prior = None
            role_prior_valid = None
            if state_role is not None:
                new_role_shape = (
                    batch_size, num_new, self.num_refines, 1)
                role_prior = torch.cat([
                    state_role, route_points.new_zeros(new_role_shape)
                ], dim=1)
                role_prior_valid = torch.cat([
                    torch.ones_like(state_role, dtype=torch.bool),
                    torch.zeros(
                        new_role_shape, device=route_points.device,
                        dtype=torch.bool)
                ], dim=1)

            teacher_role, teacher_valid = self._build_role_teacher(
                previous_match, num_new, route_points)
            role_output = self.role_router(
                conditioned_feat, route_points, base_semantics, source_flag,
                role_prior=role_prior,
                role_prior_valid=role_prior_valid,
                teacher_role=teacher_role,
                teacher_valid=teacher_valid,
                teacher_forcing_ratio=self.role_teacher_forcing_ratio)
            query_role = role_output['query_role']

            ego_next, pred_traj, predicted_pose, predicted_relative, \
                predicted_lidar_delta = self._predict_pose(
                    ego_feat, conditioned_feat, route_points, query_role,
                    geometry_cumulative, lidar_to_ego)
            pred_trajs_list.append(pred_traj)

            gt_relative = None if gt_relative_matrices is None else \
                gt_relative_matrices[:, interval]
            geometry_relative = self._select_geometry_pose(
                predicted_relative, gt_relative)
            geometry_cumulative = self.ego_warp.compose(
                geometry_cumulative, geometry_relative)

            evolution_feat = conditioned_feat + ego_next
            legacy_dynamic_xy = self.vel_branch(evolution_feat).reshape(
                batch_size, -1, self.num_refines, 2)
            evolution_output = self.dual_evolution(
                evolution_feat,
                state_points,
                new_points_t0,
                ego_next,
                query_role,
                role_output['route_role'],
                geometry_relative,
                geometry_cumulative,
                self.ego_warp,
                legacy_dynamic_xy=legacy_dynamic_xy)
            interaction_output = self.dual_interaction(
                evolution_feat, query_role,
                evolution_output['points_metric'])
            joint_output = self.joint_refine(
                evolution_feat,
                interaction_output['dynamic_feat'],
                interaction_output['static_feat'],
                evolution_output['points_metric'])
            next_feat = joint_output['query_feat']

            role_correction = joint_output['role_correction'].tanh() * \
                self.dsqe_cfg['role_correction_scale']
            corrected_role = (
                role_output['role_pred'] + role_correction).clamp(
                    self.role_router.eps, 1 - self.role_router.eps)
            corrected_role_logits = torch.logit(
                corrected_role, eps=self.role_router.eps)
            corrected_query_role = (
                role_output['pool_weights'] * corrected_role).sum(dim=2)

            correction = self.reg_branch(next_feat).reshape(
                batch_size, -1, self.num_refines, 3).tanh()
            correction = correction * self.dsqe_cfg[
                'point_correction_scale']
            correction_gate = corrected_query_role.unsqueeze(2) + \
                self.dsqe_cfg['correction_static_scale'] * \
                (1 - corrected_query_role.unsqueeze(2))
            next_points_metric = evolution_output['points_metric'] + \
                correction_gate * correction
            next_points = encode_points(next_points_metric, self.pc_range)
            next_semantics, semantic_correction = self._refine_semantics(
                base_semantics, next_feat)

            forecast_points_list.append(next_points)
            forecast_semantics_list.append(next_semantics)
            forecast_points_mask_list.append(
                next_points_metric[..., 0] >= 0)

            static_reference_metric = None
            if gt_relative is not None:
                static_reference = self.ego_warp.current_to_next(
                    state_points, gt_relative)
                static_reference_metric = decode_points(
                    static_reference, self.pc_range)

            match_cache = None
            if future_voxels is not None and hasattr(
                    self.pts_bbox_head, 'build_future_match_cache'):
                match_cache = self.pts_bbox_head.build_future_match_cache(
                    next_points, future_voxels[interval])
            match_cache_list.append(match_cache)

            dsqe_outputs.append(dict(
                source_flag=source_flag,
                num_carried=num_carried,
                role_logits=corrected_role_logits,
                role_pred=corrected_role,
                role_correction=role_correction,
                semantic_correction=semantic_correction,
                route_role=role_output['route_role'],
                query_role=corrected_query_role,
                pool_weights=role_output['pool_weights'],
                query_motion=evolution_output['query_motion'],
                carried_prior_metric=evolution_output[
                    'carried_prior_metric'],
                static_reference_metric=static_reference_metric,
                points_metric=next_points_metric,
                predicted_lidar_delta=predicted_lidar_delta,
                predicted_relative_pose=predicted_pose,
                predicted_relative_matrix=predicted_relative,
                gt_relative_pose=None if gt_relative_poses is None else
                gt_relative_poses[:, interval],
                interaction_gates=(
                    interaction_output['dynamic_from_static_gate'],
                    interaction_output['static_from_dynamic_gate']),
                match_cache=match_cache))

            state_feat = next_feat
            state_points = next_points
            state_semantics = next_semantics
            state_role = corrected_role
            activation_step = active_step
            ego_feat = ego_next
            previous_match = match_cache

        if not self.pretrain and len(pred_trajs_list) < self.num_fu_frames:
            source_flag = state_points.new_ones(
                batch_size, state_points.shape[1], 1)
            role_output = self.role_router(
                state_feat, state_points, state_semantics, source_flag,
                role_prior=state_role)
            ego_feat, extra_traj, _, _, _ = self._predict_pose(
                ego_feat, state_feat, state_points,
                role_output['query_role'], geometry_cumulative, lidar_to_ego)
            pred_trajs_list.append(extra_traj)

        return dict(
            cls_score=query_cls_all[:, current_mask],
            refine_pts=query_pos_all[:, current_mask],
            outs=outs,
            forecast_semantics_list=forecast_semantics_list,
            forecast_points_list=forecast_points_list,
            pred_trajs_list=pred_trajs_list,
            forecast_points_mask_list=forecast_points_mask_list,
            dsqe_outputs=dsqe_outputs,
            match_cache_list=match_cache_list)

    def _forward_baseline_scf(self, outs, ego_feat, img_metas, kwargs):
        ind_stamps_all = self.pts_bbox_head.ind_stamps_all
        query_feat = outs['query_feat']
        query_pos = outs['all_refine_pts'][-1]
        query_cls = outs['all_cls_scores'][-1]
        batch_size = query_feat.shape[0]
        current_mask = ind_stamps_all == 0
        current_feat = query_feat[:, current_mask]
        current_pos = query_pos[:, current_mask].detach()
        timestamp = current_pos.new_zeros(
            batch_size, self.num_query, self.num_refines, 1)

        forecast_points_list = []
        forecast_semantics_list = []
        pred_trajs_list = []
        forecast_points_mask_list = []
        if self.training:
            num_fu_frames = max(
                1, min(self.curr_epoch - self.finetune_epoch + 1,
                       self.num_fu_frames))
        else:
            num_fu_frames = self.num_fu_frames

        for interval in range(num_fu_frames):
            fused_ego_feat, _ = self.ego_cross_attn(
                ego_feat.new_full((batch_size, 1, 3), 0.5), ego_feat,
                current_pos.detach(), current_feat.detach())
            pred_trajs_list.append(self.traj_head(fused_ego_feat))
            new_mask = ind_stamps_all == interval + 1
            current_feat = torch.cat(
                [current_feat, query_feat[:, new_mask]], dim=1)
            current_pos = torch.cat(
                [current_pos, query_pos[:, new_mask]], dim=1).detach()
            new_timestamp = current_pos.new_full(
                (batch_size, int(new_mask.sum()), self.num_refines, 1), 0.5)
            timestamp = torch.cat([timestamp, new_timestamp], dim=1)
            position_embedding = self.position_encoder(
                torch.cat([current_pos, timestamp], dim=-1).flatten(2, 3))
            current_feat = current_feat + fused_ego_feat + position_embedding

            reg_offset = self.reg_branch(current_feat).reshape(
                batch_size, -1, self.num_refines, 3) * 0.5
            cls_score = self.cls_branch(current_feat).reshape(
                batch_size, -1, self.num_refines, 17)
            velocity = self.vel_branch(current_feat).reshape(
                batch_size, -1, self.num_refines, 2)
            moving_mask = ((cls_score.argmax(-1) >= 2) &
                           (cls_score.argmax(-1) <= 10)).unsqueeze(-1)
            reg_offset[..., :2] += velocity * moving_mask
            current_pos = self.refine_points(
                current_pos, reg_offset.flatten(2, 3))
            forecast_semantics_list.append(cls_score)
            forecast_points_list.append(current_pos)
            if self.training:
                forecast_points_mask_list.append(
                    self._baseline_foreground_mask(
                        current_pos, interval, img_metas, kwargs))

        if not self.pretrain and len(pred_trajs_list) < self.num_fu_frames:
            fused_ego_feat, _ = self.ego_cross_attn(
                ego_feat.new_zeros(batch_size, 1, 3), ego_feat,
                current_pos, current_feat)
            pred_trajs_list.append(self.traj_head(fused_ego_feat))

        return dict(
            cls_score=query_cls[:, current_mask],
            refine_pts=query_pos[:, current_mask],
            outs=outs,
            forecast_semantics_list=forecast_semantics_list,
            forecast_points_list=forecast_points_list,
            pred_trajs_list=pred_trajs_list,
            forecast_points_mask_list=forecast_points_mask_list,
            dsqe_outputs=None,
            match_cache_list=None)

    def forward_test(self, img_metas, img=None, **kwargs):
        if not isinstance(img_metas, list):
            raise TypeError('img_metas must be a list, but got {}'.format(
                type(img_metas)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def forward_backbone(self, img, img_metas, **kwargs):
        batch_size = img.shape[0]
        ego_feat = self._get_ego_feat(
            kwargs, batch_size, img.device, torch.float32)
        points_scale = self.points_scale_branch(ego_feat).tanh()
        self.pts_bbox_head.points_scale = (
            (points_scale + 1) / 2 * (1.5 - 0.8) + 0.8)

        if self.training:
            img_feats = self.extract_feat(img, img_metas)
            outs = self.pts_bbox_head(img_feats, img_metas)
        else:
            outs = self.simple_test_online(img_metas, img)

        if self.dsqe_enabled:
            return self._forward_dsqe_scf(outs, ego_feat, img_metas, kwargs)
        return self._forward_baseline_scf(
            outs, ego_feat, img_metas, kwargs)

    def simple_test(self, img_metas, img=None, rescale=False, **kwargs):
        for key in kwargs:
            kwargs[key] = kwargs[key][0]
        outputs = self.forward_backbone(img, img_metas, **kwargs)
        outs = outputs['outs']
        current_mask = self.pts_bbox_head.ind_stamps_all == 0
        pred_dict = dict(
            cls_scores=outs['all_cls_scores'][-1][:, current_mask],
            refine_pts=outs['all_refine_pts'][-1][:, current_mask])
        occ_pred = self.pts_bbox_head.get_occ(pred_dict)[0]
        geo_pred = torch.ones_like(occ_pred) * 17
        geo_pred[occ_pred != 17] = 0
        result = {
            'semantic_occ_0s': [occ_pred.cpu().numpy()],
            'geo_occ_0s': [geo_pred.cpu().numpy()]
        }

        for interval, (points, semantics) in enumerate(zip(
                outputs['forecast_points_list'],
                outputs['forecast_semantics_list'])):
            occ_forecast = self.pts_bbox_head.get_occ(dict(
                cls_scores=semantics, refine_pts=points))[0]
            geo_forecast = torch.ones_like(occ_forecast) * 17
            geo_forecast[occ_forecast != 17] = 0
            result.update({
                'semantic_occ_{}s'.format(interval + 1): [
                    occ_forecast.cpu().numpy()],
                'geo_occ_{}s'.format(interval + 1): [
                    geo_forecast.cpu().numpy()]
            })
        result['pred_traj'] = torch.cat(outputs['pred_trajs_list'], dim=1)
        return result

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      **kwargs):
        temporal_semantics = kwargs['temporal_semantics']
        temporal2ego = kwargs['temporal2ego']
        outputs = self.forward_backbone(img, img_metas, **kwargs)
        outs = outputs['outs']
        losses = {}
        ind_stamps_all = self.pts_bbox_head.ind_stamps_all

        loss_inputs = [
            voxel_semantics, temporal_semantics, temporal2ego, outs]
        losses.update(self.pts_bbox_head.loss_pretrain(*loss_inputs))
        if not self.pretrain:
            outs['init_points'] = None
            for index in range(len(outs['all_cls_scores'])):
                outs['all_cls_scores'][index] = \
                    outs['all_cls_scores'][index][:, ind_stamps_all == 0]
                outs['all_refine_pts'][index] = \
                    outs['all_refine_pts'][index][:, ind_stamps_all == 0]
            losses.update(self.pts_bbox_head.loss(voxel_semantics, outs))

        voxel_semantics_temporal = [
            self._dict_value(temporal_semantics, interval)['voxel_semantics']
            for interval in range(1, self.num_fu_frames + 1)
        ]
        num_fu_frames = len(outputs['forecast_semantics_list'])
        losses.update(self.pts_bbox_head.loss_future(
            voxel_semantics_temporal[:num_fu_frames],
            outputs['forecast_points_list'],
            outputs['forecast_semantics_list'],
            outputs['forecast_points_mask_list'],
            dsqe_outputs=outputs['dsqe_outputs'],
            match_cache_list=outputs['match_cache_list']))

        for interval, pred_traj in enumerate(outputs['pred_trajs_list']):
            losses.update(self.loss_traj(
                pred_traj.squeeze(1),
                kwargs['temporal_trajs'][:, interval, :], interval + 1))
        return losses
