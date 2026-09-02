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
from mmdet3d.utils import get_root_logger
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
    # The model returns temporal occupancy/trajectory dictionaries rather
    # than the generic detector result list expected by MMDetection hooks.
    uses_sparseworld_eval_api = True

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
            self._init_dsqe(drop_out)
            self._configure_dsqe_stage()

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
        cfg.setdefault('semantic_residual_scale', 0.5)
        cfg.setdefault('role_correction_scale', 0.25)
        cfg.setdefault('stream_dropout', 0.1)
        cfg.setdefault('dsqe_training_stage', 'residual_stage1')
        cfg.setdefault('freeze_baseline', True)
        cfg.setdefault('freeze_tass', True)
        cfg.setdefault('planning_gradient_to_dsqe', False)
        cfg.setdefault('forecast_curriculum_enabled', False)
        cfg.setdefault('forecast_curriculum_start_epoch', 0)
        cfg.setdefault('feature_residual_scale', 1.0)
        cfg.setdefault('dynamic_point_delta_scale', 1.0)
        cfg.setdefault('static_point_delta_scale', 0.2)

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
        self.feature_residual_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim))

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
        self.pts_bbox_head.freeze_tass = bool(cfg.get('freeze_tass', True))

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
            self.feature_residual_head[-1].weight.zero_()
            self.feature_residual_head[-1].bias.zero_()

    def _configure_dsqe_stage(self):
        """Freeze the fixed BaseLine carrier for residual stage-1 training."""
        cfg = self.dsqe_cfg
        self.pts_bbox_head.freeze_tass = bool(cfg.get('freeze_tass', True))
        if not cfg.get('freeze_baseline', True):
            return
        baseline_modules = [
            self.img_backbone, self.img_neck, self.pts_bbox_head,
            self.plan_head, self.points_scale_branch, self.ego_cross_attn,
            self.position_encoder,
            self.reg_branch, self.vel_branch, self.cls_branch, self.traj_head,
        ]
        for module in baseline_modules:
            if module is not None:
                module.requires_grad_(False)
        # Legacy new-query point update is retained only for old callers;
        # residual stage-1 uses the dynamic/static heads for every query.
        self.dual_evolution.new_update_head.requires_grad_(False)

    def _parameter_group_summary(self):
        groups = {}
        for name, parameter in self.named_parameters():
            group = name.split('.')[0]
            state = 'trainable' if parameter.requires_grad else 'frozen'
            groups.setdefault(group, {'trainable': 0, 'frozen': 0})[state] += \
                parameter.numel()
        return groups

    def init_weights(self):
        self.pts_bbox_head.init_weights()
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)
        if self.dsqe_enabled:
            self.ego_cross_attn.init_weights()
            self.ego_cross_attn_static.init_weights()
            total = sum(parameter.numel() for parameter in self.parameters())
            trainable = sum(parameter.numel() for parameter in self.parameters()
                            if parameter.requires_grad)
            logger = get_root_logger()
            logger.info('DSQE stage=%s total_params=%d trainable_params=%d',
                        self.dsqe_cfg.get('dsqe_training_stage'), total,
                        trainable)
            for name, values in self._parameter_group_summary().items():
                logger.info('DSQE params %s trainable=%d frozen=%d', name,
                            values['trainable'], values['frozen'])

    def set_epoch(self, epoch):
        self.curr_epoch = epoch
        if not self.dsqe_enabled:
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
            return

        # DSQE residual stage never re-enters the BaseLine pretrain routine;
        # future-step curriculum, if desired, is controlled independently.
        self.pretrain = False
        self.pts_bbox_head.pretrain = False
        self._ensure_tass_state()

        if self.dsqe_enabled:
            start = self.dsqe_cfg['teacher_forcing_start_epoch']
            end = self.dsqe_cfg['teacher_forcing_end_epoch']
            self.role_teacher_forcing_ratio = _linear_schedule(
                self.dsqe_cfg['role_teacher_forcing'], epoch, start, end)
            self.ego_teacher_forcing_ratio = _linear_schedule(
                self.dsqe_cfg['ego_teacher_forcing'], epoch, start, end)

    def _ensure_tass_state(self):
        """Initialize and synchronize the fixed TASS assignment once."""
        head = self.pts_bbox_head
        if getattr(head, 'ind_stamps_all', None) is None:
            with torch.no_grad():
                weights = head.num_stamps_all.float()
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1)
                head.ind_stamps_all = get_matched_inds(
                    weights, [self.num_query] + self.num_fu_query)
            head.reset_mask()

        freeze_tass = self.dsqe_enabled and self.dsqe_cfg.get(
            'freeze_tass', True)
        if freeze_tass and not hasattr(self, '_tass_frozen_num_stamps'):
            self._tass_frozen_num_stamps = head.num_stamps_all.detach().clone()
            self._tass_frozen_ind_stamps = head.ind_stamps_all.detach().clone()
        if freeze_tass:
            head.num_stamps_all.copy_(self._tass_frozen_num_stamps)
            head.ind_stamps_all = self._tass_frozen_ind_stamps.clone()
        expected_counts = [self.num_query] + list(self.num_fu_query)
        actual_counts = torch.bincount(
            head.ind_stamps_all.long(), minlength=len(expected_counts)).tolist()
        if actual_counts != expected_counts:
            raise RuntimeError(
                'TASS slot counts mismatch: expected {}, got {}'.format(
                    expected_counts, actual_counts))
        self._synchronize_tass_state()

    def _synchronize_tass_state(self):
        """Broadcast TASS state and fail loudly on rank divergence."""
        head = self.pts_bbox_head
        distributed = torch.distributed.is_available() and \
            torch.distributed.is_initialized()
        difference = 0
        if distributed:
            local_num = head.num_stamps_all.detach().clone()
            root_num = local_num.clone()
            torch.distributed.broadcast(root_num, src=0)
            if not torch.equal(local_num, root_num):
                raise RuntimeError('TASS num_stamps_all mismatch across ranks')
            local_stamps = head.ind_stamps_all.to(
                head.num_stamps_all.device).detach().clone()
            root_stamps = local_stamps.clone()
            torch.distributed.broadcast(root_stamps, src=0)
            difference = int((local_stamps != root_stamps).sum().item())
            head.tass_rank_difference = difference
            if difference:
                raise RuntimeError(
                    'TASS ind_stamps_all mismatch across ranks: {}'.format(
                        difference))
            head.num_stamps_all.copy_(root_num)
            head.ind_stamps_all = root_stamps
        checksum_num = int(head.num_stamps_all.long().sum().item())
        checksum_ind = int(head.ind_stamps_all.long().sum().item())
        head.tass_checksum_num = checksum_num
        head.tass_checksum_ind = checksum_ind
        if not distributed:
            head.tass_rank_difference = 0
        rank = torch.distributed.get_rank() if distributed else 0
        get_root_logger().info(
            'TASS rank=%d num_stamps_checksum=%d ind_stamps_checksum=%d '
            'rank_difference=%d', rank, checksum_num, checksum_ind,
            head.tass_rank_difference)
        head.reset_mask()

    def _num_forecast_frames(self):
        if not self.training:
            return self.num_fu_frames
        if not self.dsqe_enabled:
            return max(1, min(self.curr_epoch - self.finetune_epoch + 1,
                              self.num_fu_frames))
        if not self.dsqe_cfg.get('forecast_curriculum_enabled', False):
            return self.num_fu_frames
        start = int(self.dsqe_cfg.get('forecast_curriculum_start_epoch', 0))
        return max(1, min(self.curr_epoch - start + 1,
                          self.num_fu_frames))

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
        # ``DataContainer(stack=False)`` values are intentionally kept as a
        # per-sample list by the dataloader.  This helper is used for fields
        # that are guaranteed to have a rectangular batch shape (poses,
        # ego-state tensors, etc.); variable-length actor fields use
        # ``_to_batch_sequence`` below instead of being stacked here.
        if hasattr(value, 'data') and not isinstance(value, torch.Tensor):
            value = value.data

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

    @staticmethod
    def _to_batch_sequence(value, device, dtype):
        """Convert a possibly ragged field into one tensor per sample.

        Actor boxes and actor attributes have a variable first dimension (the
        number of actors in a scene), so they must not be passed through
        ``torch.stack``.  The returned list is deliberately kept ragged and
        is consumed sample-by-sample by the role target builder.
        """
        if value is None:
            return None
        if hasattr(value, 'data') and not isinstance(value, torch.Tensor):
            value = value.data

        # mmcv's ragged collate stores one non-stacked field as
        # ``[sample_0, sample_1, ...]`` inside an outer micro-batch list.
        # Unwrap that container before converting individual samples.
        if isinstance(value, (list, tuple)) and len(value) == 1 and \
                isinstance(value[0], (list, tuple)):
            value = value[0]

        def _convert(item):
            if hasattr(item, 'data') and not isinstance(item, torch.Tensor):
                item = item.data
            if isinstance(item, torch.Tensor):
                tensor = item.to(device=device, dtype=dtype)
            else:
                tensor = torch.as_tensor(
                    np.asarray(item), device=device, dtype=dtype)
            # A single-sample DataContainer may retain a leading singleton
            # dimension.  Remove only those dimensions, never actor rows.
            while tensor.ndim > 2 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            return tensor

        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device, dtype=dtype)
            if tensor.ndim <= 2:
                return [_convert(tensor)]
            return [_convert(tensor[index]) for index in range(tensor.shape[0])]
        if isinstance(value, (list, tuple)):
            if not value:
                return []
            # A raw unbatched list of actor rows is one sample; a collated
            # ragged batch is a list whose elements are matrices.
            def _ndim(item):
                if isinstance(item, torch.Tensor):
                    return item.ndim
                return np.asarray(item, dtype=object).ndim
            if all(_ndim(item) <= 1 for item in value):
                return [_convert(value)]
            return [_convert(item) for item in value]
        return [_convert(value)]

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

    def _build_role_metadata(self, kwargs, interval,
                             gt_relative_matrices=None):
        """Build motion-state actor metadata for role supervision.

        The dataset provides current actor boxes (including velocity) and
        future actor attributes.  We use those attributes when available and
        compensate the current center with the GT ego transform.  Matching is
        deliberately conservative: points outside an actor footprint remain
        invalid instead of being assigned a semantic-class pseudo label.
        """
        boxes = kwargs.get('temporal_agent_boxes')
        feats = kwargs.get('temporal_agent_feats')
        if boxes is None:
            return None
        device = gt_relative_matrices.device if isinstance(
            gt_relative_matrices, torch.Tensor) else \
            (boxes.device if isinstance(boxes, torch.Tensor) else
             torch.device('cpu'))
        dtype = boxes.dtype if isinstance(boxes, torch.Tensor) else torch.float32
        boxes = self._to_batch_sequence(boxes, device, dtype)
        feats = self._to_batch_sequence(feats, device, dtype) \
            if feats is not None else None
        if not boxes:
            return []
        batch_size = len(boxes)
        metadata = []
        tau = float(self.dsqe_cfg.get('role_speed_threshold', 0.5))
        temperature = float(self.dsqe_cfg.get('role_speed_temperature', 0.1))
        dt = float(self.dsqe_cfg.get('role_frame_dt', 0.5))
        for batch_index in range(batch_size):
            current = boxes[batch_index]
            if current.ndim == 1:
                current = current.unsqueeze(0)
            num_agents = current.shape[0]
            centers = current[:, :3]
            if current.shape[-1] >= 9:
                velocity = current[:, 7:9]
            else:
                velocity = centers.new_zeros(num_agents, 2)
            future_centers = centers.clone()
            future_valid = torch.ones(num_agents, dtype=torch.bool,
                                       device=centers.device)
            attribute_valid = future_valid.clone()
            trajectory_available = False
            trajectory_speed = None
            future_yaw_delta = None
            if feats is not None and batch_index < len(feats):
                agent_feat = feats[batch_index]
                if agent_feat.ndim == 1:
                    agent_feat = agent_feat.unsqueeze(0)
                # Keep the actor/attribute pairing well-defined even if an
                # upstream sample contains a malformed trailing row.
                num_attr_agents = min(num_agents, agent_feat.shape[0])
                if num_attr_agents != num_agents:
                    padded = centers.new_zeros(
                        num_agents, agent_feat.shape[-1])
                    padded[:num_attr_agents] = agent_feat[:num_attr_agents]
                    agent_feat = padded
                    attribute_valid[num_attr_agents:] = False
                trajectory_dims = 2 * self.num_fu_frames
                if agent_feat.shape[-1] >= trajectory_dims:
                    trajectory_available = True
                    trajectories = agent_feat[:, :trajectory_dims].reshape(
                        num_agents, self.num_fu_frames, 2)
                    step = max(0, min(interval - 1, self.num_fu_frames - 1))
                    step_displacement = trajectories[:, step]
                    cumulative_displacement = trajectories[:, :step + 1].sum(
                        dim=1)
                    trajectory_speed = step_displacement.norm(dim=-1) / max(
                        dt, 1e-3)
                    # Interval zero is matched against current-frame
                    # occupancy, so its actor centers must remain current.
                    if interval > 0:
                        # VAD-style future trajectories are adjacent-frame
                        # increments.  Future actor centers therefore use
                        # the cumulative displacement through this interval,
                        # not only the final segment.
                        future_centers[:, :2] = centers[:, :2] + \
                            cumulative_displacement
                    mask_start = trajectory_dims
                    if agent_feat.shape[-1] >= mask_start + self.num_fu_frames:
                        future_valid = agent_feat[:, mask_start + step] > 0.5
                    # VAD stores future yaw as adjacent-frame yaw deltas at
                    # the end of the attribute vector, just like the future
                    # xy trajectories.  Keep the full vector so the yaw can
                    # be accumulated through the selected interval.
                    yaw_start = agent_feat.shape[-1] - self.num_fu_frames
                    if yaw_start >= mask_start + self.num_fu_frames + 1:
                        future_yaw_delta = agent_feat[
                            :, yaw_start:yaw_start + self.num_fu_frames]
            future_valid = future_valid & attribute_valid

            compensated_current = centers
            if gt_relative_matrices is not None and interval > 0 and \
                    interval - 1 < gt_relative_matrices.shape[1]:
                # Relative matrices are next->current; invert to map current
                # actor centers into the future ego frame.
                transform = self.ego_warp.identity(
                    1, centers.device, centers.dtype)[0]
                for step_index in range(min(interval,
                                            gt_relative_matrices.shape[1])):
                    transform = torch.matmul(
                        transform, gt_relative_matrices[batch_index, step_index])
                future_transform = self.ego_warp.inverse(transform.unsqueeze(0))
                compensated_current = self.ego_warp.transform_metric(
                    centers.unsqueeze(0), future_transform)[0]
                future_centers = self.ego_warp.transform_metric(
                    future_centers.unsqueeze(0),
                    future_transform)[0]

            displacement = future_centers[:, :2] - compensated_current[:, :2]
            speed = displacement.norm(dim=-1) / max(
                dt * max(interval, 1), 1e-3)
            if trajectory_speed is not None:
                # Position matching uses cumulative displacement, whereas
                # the role at this forecast step represents the actor's
                # instantaneous adjacent-frame motion state.
                speed = trajectory_speed
            # If no future attribute is available, current velocity is the
            # explicit fallback; it is still a motion-state estimate, not a
            # semantic class heuristic.
            if not trajectory_available:
                speed = velocity.norm(dim=-1)
            role = torch.sigmoid((speed - tau) / max(temperature, 1e-3))
            if current.shape[-1] >= 6:
                # Half the BEV box diagonal covers all points in the actor
                # footprint while retaining an explicit distance threshold.
                radius = 0.5 * current[:, 3:5].abs().square().sum(
                    dim=-1).sqrt()
            else:
                radius = centers.new_full((num_agents,), 1.0)
            actor_dims = current[:, 3:6].abs() \
                if current.shape[-1] >= 6 else None
            actor_yaw = current[:, 6] \
                if current.shape[-1] >= 7 else None
            if actor_yaw is not None and future_yaw_delta is not None and \
                    interval > 0:
                yaw_step = min(interval - 1, future_yaw_delta.shape[1] - 1)
                actor_yaw = actor_yaw + future_yaw_delta[:, :yaw_step + 1].sum(
                    dim=1)
            if actor_yaw is not None and gt_relative_matrices is not None and \
                    interval > 0 and interval - 1 < gt_relative_matrices.shape[1]:
                # Rotate the actor footprint with the same current->future
                # transform used for its center.
                future_yaw = torch.atan2(
                    future_transform[:, 1, 0],
                    future_transform[:, 0, 0])[0]
                actor_yaw = actor_yaw + future_yaw
            metadata.append(dict(
                centers=future_centers,
                radius=radius.clamp_min(0.5),
                dims=actor_dims,
                yaw=actor_yaw,
                role=role,
                valid=future_valid))
        return metadata

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
        if target is None:
            return predicted
        # Explicit evaluation-only oracle for the pose diagnostic.  Normal
        # validation keeps using the predicted pose.
        if (not self.training and
                self.dsqe_cfg.get('eval_oracle_pose', False)):
            return target
        if not self.training or self.ego_teacher_forcing_ratio <= 0:
            return predicted
        batch_size = predicted.shape[0]
        use_target = torch.rand(
            batch_size, device=predicted.device) < \
            self.ego_teacher_forcing_ratio
        return torch.where(use_target[:, None, None], target, predicted)

    def _build_role_teacher(self, previous_match, num_new, reference,
                            force=False):
        teacher_ratio = 1.0 if force else self.role_teacher_forcing_ratio
        if previous_match is None or teacher_ratio <= 0:
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

    def _refine_semantics(self, base_semantics, next_feat,
                          baseline_feat=None):
        """Add an independent per-step semantic residual.

        ``baseline_feat`` is deliberately separate from ``next_feat`` so the
        legacy classification head can only consume the BaseLine feature
        distribution.
        """
        residual_feat = next_feat
        baseline_feat = next_feat if baseline_feat is None else baseline_feat
        batch_size = baseline_feat.shape[0]
        if (not getattr(self, 'training', True) and
                self.dsqe_cfg.get('eval_legacy_semantics', False)):
            # Keep DSQE point evolution fixed while replacing only the
            # semantic output with the legacy per-step cls branch.
            return self.cls_branch(baseline_feat).reshape(
                batch_size, -1, self.num_refines, 17), None
        correction = self.semantic_correction_head(residual_feat).reshape(
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
        # DSQE state is allowed to carry residual-corrected values.  The
        # separate BaseLine carrier below is never replaced by this state.
        state_feat = query_feat_all[:, current_mask]
        state_points = query_pos_all[:, current_mask].detach()
        baseline_feat = state_feat
        baseline_points = state_points
        baseline_timestamp = state_points.new_zeros(
            batch_size, self.num_query, self.num_refines, 1)
        state_role = None
        activation_step = torch.zeros(
            batch_size, state_feat.shape[1], device=state_feat.device,
            dtype=torch.long)

        oracle_role = (not self.training and
                       self.dsqe_cfg.get('eval_oracle_role', False))
        oracle_pose = (not self.training and
                       self.dsqe_cfg.get('eval_oracle_pose', False))
        if self.training or oracle_role or oracle_pose:
            gt_relative_matrices, gt_relative_poses = self._get_gt_ego_poses(
                img_metas, kwargs, state_feat.device, state_feat.dtype)
        else:
            # Ordinary inference must not consume future GT poses.
            gt_relative_matrices, gt_relative_poses = None, None
        lidar_to_ego = self._get_lidar_to_ego(
            img_metas, state_feat.device, state_feat.dtype)
        future_voxels = self._get_future_voxels(kwargs) \
            if (self.training or oracle_role) else None
        geometry_cumulative = self.ego_warp.identity(
            batch_size, state_feat.device, state_feat.dtype)
        dsqe_ego_feat = ego_feat

        forecast_points_list = []
        forecast_semantics_list = []
        pred_trajs_list = []
        forecast_points_mask_list = []
        dsqe_outputs = []
        match_cache_list = []
        previous_match = None

        # Seed the first oracle-role step with labels matched against the
        # current-frame occupancy.  Later steps use the cache from the
        # preceding forecast.  New slots remain invalid in the teacher mask
        # and therefore continue to use their semantic prior.
        if oracle_role:
            current_voxel = kwargs.get('voxel_semantics')
            if current_voxel is not None:
                previous_match = \
                    self.pts_bbox_head.build_future_match_cache(
                        state_points, current_voxel,
                        role_metadata=self._build_role_metadata(
                            kwargs, 0, gt_relative_matrices))

        num_fu_frames = self._num_forecast_frames()

        for interval in range(num_fu_frames):
            baseline_step = SparseWorld4DTraj._baseline_future_step(
                self, baseline_feat, baseline_points, baseline_timestamp,
                ego_feat, query_feat_all, query_pos_all, query_cls_all,
                ind_stamps_all, interval)
            baseline_feat = baseline_step['feat']
            baseline_points = baseline_step['points']
            baseline_timestamp = baseline_step['timestamp']
            base_next_semantics = baseline_step['semantics']
            new_mask = baseline_step['new_mask']
            num_carried = state_feat.shape[1]
            num_new = int(new_mask.sum())
            base_next_metric = decode_points(
                baseline_points, self.pc_range)
            route_points = torch.cat([state_points, baseline_points[:, num_carried:]], dim=1)
            base_feat = torch.cat([
                state_feat, baseline_feat[:, num_carried:]], dim=1)
            base_semantics = base_next_semantics
            source_flag = torch.cat([
                route_points.new_ones(batch_size, num_carried, 1),
                route_points.new_zeros(batch_size, num_new, 1)
            ], dim=1)
            new_activation = torch.full(
                (batch_size, num_new), interval + 1,
                device=activation_step.device, dtype=torch.long)
            active_step = torch.cat(
                [activation_step, new_activation], dim=1)
            conditioned_feat = base_feat + \
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

            if oracle_role and future_voxels is not None:
                oracle_cache = self.pts_bbox_head.build_future_match_cache(
                    route_points, future_voxels[interval],
                    role_metadata=self._build_role_metadata(
                        kwargs, interval + 1, gt_relative_matrices))
                teacher_role = oracle_cache['role_target'].unsqueeze(-1)
                teacher_valid = oracle_cache['role_valid'].unsqueeze(-1)
            else:
                teacher_role, teacher_valid = self._build_role_teacher(
                    previous_match, num_new, route_points, force=False)
            role_output = self.role_router(
                conditioned_feat, route_points, base_semantics, source_flag,
                role_prior=role_prior,
                role_prior_valid=role_prior_valid,
                teacher_role=teacher_role,
                teacher_valid=teacher_valid,
                teacher_forcing_ratio=(1.0 if oracle_role else
                                       self.role_teacher_forcing_ratio))
            query_role = role_output['query_role']

            ego_next, _, predicted_pose, predicted_relative, \
                predicted_lidar_delta = self._predict_pose(
                    dsqe_ego_feat, conditioned_feat, route_points, query_role,
                    geometry_cumulative, lidar_to_ego)
            # Planning remains the BaseLine trajectory in residual stage-1.
            pred_trajs_list.append(baseline_step['pred_traj'])

            gt_relative = None if gt_relative_matrices is None else \
                gt_relative_matrices[:, interval]
            geometry_relative = self._select_geometry_pose(
                predicted_relative, gt_relative)
            geometry_cumulative = self.ego_warp.compose(
                geometry_cumulative, geometry_relative)

            evolution_feat = conditioned_feat + ego_next
            new_points_t0 = query_pos_all[:, new_mask].detach()
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
                base_points=baseline_points)
            interaction_output = self.dual_interaction(
                evolution_feat, query_role,
                base_next_metric)
            joint_output = self.joint_refine(
                evolution_feat,
                interaction_output['dynamic_feat'],
                interaction_output['static_feat'],
                base_next_metric)

            role_correction = joint_output['role_correction'].tanh() * \
                self.dsqe_cfg['role_correction_scale']
            role_base = role_output['route_role'] if oracle_role else \
                role_output['role_pred']
            corrected_role = role_base if oracle_role else (
                role_base + role_correction).clamp(
                    self.role_router.eps, 1 - self.role_router.eps)
            corrected_role_logits = torch.logit(
                corrected_role, eps=self.role_router.eps)
            corrected_query_role = (
                role_output['pool_weights'] * corrected_role).sum(dim=2)

            dynamic_delta = evolution_output['dynamic_residual']
            static_delta = evolution_output['static_residual']
            dynamic_delta = dynamic_delta * self.dsqe_cfg.get(
                'dynamic_point_delta_scale', 1.0)
            static_delta = static_delta * self.dsqe_cfg.get(
                'static_point_delta_scale', 0.2)
            next_delta = corrected_role * dynamic_delta + \
                (1 - corrected_role) * static_delta
            next_points_metric = base_next_metric + next_delta
            next_points = encode_points(next_points_metric, self.pc_range)
            delta_feat = self.feature_residual_head(
                joint_output['query_feat']) * self.dsqe_cfg.get(
                    'feature_residual_scale', 1.0)
            # Feature residuals are anchored to the complete BaseLine carrier,
            # not to the previous DSQE-corrected state.
            next_feat = baseline_feat + delta_feat
            next_semantics, semantic_correction = self._refine_semantics(
                base_next_semantics, joint_output['query_feat'],
                baseline_feat)

            forecast_points_list.append(next_points)
            forecast_semantics_list.append(next_semantics)
            if self.training:
                # Reuse the official BaseLine foreground mask. With zero
                # DSQE residual this exactly matches the DSQE-off path.
                forecast_points_mask_list.append(
                    self._baseline_foreground_mask(
                        next_points, interval, img_metas, kwargs))

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
                    next_points, future_voxels[interval],
                    role_metadata=self._build_role_metadata(
                        kwargs, interval + 1, gt_relative_matrices))
            match_cache_list.append(match_cache)

            dsqe_outputs.append(dict(
                source_flag=source_flag,
                num_carried=num_carried,
                role_logits=corrected_role_logits,
                role_pred=corrected_role,
                role_correction=role_correction,
                semantic_correction=semantic_correction,
                delta_feat=delta_feat,
                delta_points=next_delta,
                base_feat=baseline_feat,
                base_points=baseline_points,
                base_semantics=base_next_semantics,
                dynamic_delta=dynamic_delta,
                static_delta=static_delta,
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
            state_role = corrected_role
            activation_step = active_step
            dsqe_ego_feat = ego_next
            previous_match = match_cache
            # Only the BaseLine carrier advances through the original step.

        if not self.pretrain and len(pred_trajs_list) < self.num_fu_frames:
            fused_ego_feat, _ = self.ego_cross_attn(
                ego_feat.new_zeros(batch_size, 1, 3), ego_feat,
                baseline_points, baseline_feat)
            pred_trajs_list.append(self.traj_head(fused_ego_feat))

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

    def _baseline_future_step(self, current_feat, current_pos, timestamp,
                              ego_feat, query_feat, query_pos, query_cls,
                              ind_stamps_all, interval):
        """Run exactly one original SparseWorld future update.

        This is the carrier used by both the DSQE-off path and the DSQE
        residual path.  In particular, the BaseLine ``cls_branch`` only ever
        sees the BaseLine-conditioned feature tensor produced here.
        """
        batch_size = current_feat.shape[0]
        fused_ego_feat, _ = self.ego_cross_attn(
            ego_feat.new_full((batch_size, 1, 3), 0.5), ego_feat,
            current_pos.detach(), current_feat.detach())
        pred_traj = self.traj_head(fused_ego_feat)
        new_mask = ind_stamps_all == interval + 1
        next_feat = torch.cat(
            [current_feat, query_feat[:, new_mask]], dim=1)
        next_pos = torch.cat(
            [current_pos, query_pos[:, new_mask]], dim=1).detach()
        new_timestamp = next_pos.new_full(
            (batch_size, int(new_mask.sum()), self.num_refines, 1), 0.5)
        next_timestamp = torch.cat([timestamp, new_timestamp], dim=1)
        position_embedding = self.position_encoder(
            torch.cat([next_pos, next_timestamp], dim=-1).flatten(2, 3))
        next_feat = next_feat + fused_ego_feat + position_embedding

        reg_offset = self.reg_branch(next_feat).reshape(
            batch_size, -1, self.num_refines, 3) * 0.5
        next_semantics = self.cls_branch(next_feat).reshape(
            batch_size, -1, self.num_refines, 17)
        velocity = self.vel_branch(next_feat).reshape(
            batch_size, -1, self.num_refines, 2)
        moving_mask = ((next_semantics.argmax(-1) >= 2) &
                       (next_semantics.argmax(-1) <= 10)).unsqueeze(-1)
        reg_offset[..., :2] += velocity * moving_mask
        next_pos = self.refine_points(
            next_pos, reg_offset.flatten(2, 3))
        return dict(
            feat=next_feat,
            points=next_pos,
            timestamp=next_timestamp,
            semantics=next_semantics,
            pred_traj=pred_traj,
            new_mask=new_mask,
            fused_ego_feat=fused_ego_feat)

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
        if hasattr(self, '_num_forecast_frames'):
            num_fu_frames = self._num_forecast_frames()
        else:
            num_fu_frames = max(
                1, min(self.curr_epoch - self.finetune_epoch + 1,
                       self.num_fu_frames))

        for interval in range(num_fu_frames):
            step = SparseWorld4DTraj._baseline_future_step(
                self, current_feat, current_pos, timestamp, ego_feat,
                query_feat, query_pos, query_cls, ind_stamps_all, interval)
            current_feat = step['feat']
            current_pos = step['points']
            timestamp = step['timestamp']
            pred_trajs_list.append(step['pred_traj'])
            forecast_semantics_list.append(step['semantics'])
            forecast_points_list.append(step['points'])
            if self.training:
                forecast_points_mask_list.append(
                    self._baseline_foreground_mask(
                        step['points'], interval, img_metas, kwargs))

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
        if self.dsqe_enabled and getattr(
                self.pts_bbox_head, 'ind_stamps_all', None) is None:
            # The training hook normally initializes this state.  Keep a
            # direct eval/CPU invocation equally deterministic across ranks.
            self._ensure_tass_state()
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
            if self.dsqe_enabled and not self.dsqe_cfg.get(
                    'planning_gradient_to_dsqe', False):
                pred_traj_for_loss = pred_traj.detach()
            else:
                pred_traj_for_loss = pred_traj
            losses.update(self.loss_traj(
                pred_traj_for_loss.squeeze(1),
                kwargs['temporal_trajs'][:, interval, :], interval + 1))
        return losses
