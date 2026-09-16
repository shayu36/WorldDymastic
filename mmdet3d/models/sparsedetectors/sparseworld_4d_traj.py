# Copyright (c) Phigent Robotics. All rights reserved.
from mmdet3d.models.detectors.bevdet_occ import BEVStereo4DOCC
from .opus import OPUS
import torch.nn.functional as F
import torch
import time
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.cnn.bricks.transformer import MultiheadAttention
from torch import nn
import numpy as np
from mmdet3d.models import builder
from .opus_transformer import OPUSSelfAttention, OPUSCrossAttention
from mmcv.cnn import bias_init_with_prob
from mmdet3d.models.detectors.loss import CE_ssc_loss, sem_scal_loss, geo_scal_loss, l1_loss, l2_loss
from mmdet3d.models.detectors.lovasz_softmax import lovasz_softmax
from IPython import embed
from mmdet3d.models.sparsedetectors.bbox.utils import decode_points, encode_points, trans_coords,get_matched_inds
from mmdet3d.models.heads import DownScaleModule3DCustom
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
device = torch.device('cuda')
# occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947,
                                   497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031,
                                   141625221, 2307405309])
import time
# from ptflops import get_model_complexity_info
from thop import profile

def Scatter(src_dict):
    for key, value in src_dict.items():
        if isinstance(value, torch.Tensor):
            src_dict[key] = value.cuda()
        if isinstance(value, dict):
            src_dict[key] = Scatter(value)
        if isinstance(value, list):
            if isinstance(value[0], dict):
                src_dict[key] = [Scatter(v) for v in value]
            if isinstance(value[0], torch.Tensor):
                src_dict[key] = [v.cuda() for v in value]
    return src_dict


@DETECTORS.register_module()
class SparseWorld4DTraj(OPUS):
    # Temporal occupancy/planning dictionaries use the dedicated collector.
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
                 finetune_epoch = 0,
                 num_out_query=600,
                 empty_idx=17,
                 use_focal_loss=True,
                 balance_cls_weight=True,
                 final_softplus=True,
                 dsqe_cfg=None,
                 dsqe_mode='baseline',
                 **kwargs):
        super(SparseWorld4DTraj, self).__init__(**kwargs)
        self.dataset_type = dataset_type
        self.out_dim = out_dim
        self.use_3d_loss = use_3d_loss
        self.test_threshold = test_threshold
        self.num_refines = self.pts_bbox_head.transformer.num_refines[-1]
        self.balance_cls_weight = balance_cls_weight
        self.final_softplus = final_softplus
        # self.if_pretrain = if_pretrain
        self.if_render = if_render
        self.if_post_finetune = if_post_finetune
        self.empty_idx = empty_idx
        if self.balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
            self.semantic_loss = nn.CrossEntropyLoss(
                weight=self.class_weights, reduction="mean"
            )
        else:
            self.semantic_loss = nn.CrossEntropyLoss(reduction="mean")

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
            nn.Linear(256, self.out_dim)
        )
        self.ego_cross_attn = OPUSCrossAttention(self.out_dim, 8, drop_out, self.pts_bbox_head.pc_range)

        self.position_encoder = nn.Sequential(
            nn.Linear(4 * self.num_refines, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True),
        )

        self.reg_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 3)
        )

        self.vel_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 2)
        )

        self.cls_branch = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.num_refines * 17)
        )
        self.points_scale_branch = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,3),
        )

        self.traj_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, 2),
        )
        self.l2_loss = l2_loss()

        self.box_mode_3d = Box3DMode.LIDAR
        self.planning_metric = None
        self.finetune_epoch = finetune_epoch
        self.curr_epoch = 0
        self.dsqe_cfg = dict(dsqe_cfg or {})
        self.dsqe_mode = (self.dsqe_cfg.get('mode', dsqe_mode)
                          if self.dsqe_cfg.get('enabled', True) else 'baseline')
        self.dsqe_enabled = self.dsqe_mode == 'prescf'
        if self.dsqe_mode not in ('baseline', 'prescf'):
            raise ValueError("dsqe_mode must be 'baseline' or 'prescf'")

        self.register_buffer('pred_num', torch.zeros(18), persistent=False)

        self.gt_traj = list()
        self.tau = list()
        if self.dsqe_mode == 'prescf':
            self._init_prescf(drop_out)

    def _init_prescf(self, drop_out):
        # Lazy imports keep the reference BaseLine construction free of
        # PreSCF parameters and optional DSQE dependencies.
        from .dsqe_role_router import DSQERoleRouter
        from .dsqe_ego_warp import DSQEEgoWarp
        from .dsqe_dual_evolution import DSQEDualEvolution
        from .dsqe_dual_interaction import DSQEDualInteraction
        from .dsqe_joint_refine import DSQEJointRefine
        cfg = self.dsqe_cfg
        self.pts_bbox_head.dsqe_cfg = cfg
        self.dynamic_class_ids = tuple(cfg.get(
            'dynamic_class_ids', [2, 3, 4, 5, 6, 7, 9, 10]))
        self.static_class_ids = tuple(cfg.get(
            'static_class_ids', [1, 8, 11, 12, 13, 14, 15, 16]))
        self.prescf_actor_recovery_max_distance = float(cfg.get(
            'actor_recovery_max_distance',
            cfg.get('role_match_max_distance', 2.5)))
        if self.prescf_actor_recovery_max_distance <= 0:
            raise ValueError('actor_recovery_max_distance must be positive')
        self.role_router = DSQERoleRouter(
            self.out_dim, num_classes=17,
            dynamic_class_ids=self.dynamic_class_ids,
            hidden_dims=cfg.get('role_hidden_dims', 64))
        self.ego_warp = DSQEEgoWarp(
            self.pc_range, frame_mode=cfg.get('frame_mode', 'future_ego'))
        self.dual_evolution = DSQEDualEvolution(
            self.out_dim, self.num_refines, self.pc_range,
            motion_scale=cfg.get('motion_scale', 4.0),
            static_alpha=cfg.get('static_alpha', 0.1),
            residual_scale=cfg.get('point_residual_scale', 1.0),
            new_residual_scale=cfg.get('new_residual_scale', 1.0),
            planar_motion_only=cfg.get('planar_motion_only', False))
        self.dual_interaction = DSQEDualInteraction(
            self.out_dim, num_heads=cfg.get('num_heads', 8),
            local_k=cfg.get('local_k', 16),
            dropout=cfg.get('dropout', drop_out),
            dynamic_from_static_init=cfg.get('lambda_DS', 1.0),
            static_from_dynamic_init=cfg.get('lambda_SD', 0.25))
        self.joint_refine = DSQEJointRefine(
            self.out_dim, self.num_refines, num_classes=17,
            num_heads=cfg.get('num_heads', 8),
            local_k=cfg.get('joint_local_k', cfg.get('local_k', 16)))
        self.source_embedding = nn.Embedding(2, self.out_dim)
        self.activation_embedding = nn.Embedding(
            self.num_fu_frames + 1, self.out_dim)
        # These embeddings annotate the state provenance/time slot but must
        # not perturb a loaded BaseLine representation at initialization.
        nn.init.zeros_(self.source_embedding.weight)
        nn.init.zeros_(self.activation_embedding.weight)
        self.prescf_ego_pose_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim), nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, 4))
        # Planning reads the dynamic and static Query views separately.  The
        # fused context below is also used by the ego-pose head.  Planning is
        # derived from that same pose translation after conversion to VAD's
        # fixed E0-LiDAR convention; there is no second trajectory head.
        self.ego_cross_attn_dynamic = OPUSCrossAttention(
            self.out_dim, 8, drop_out, self.pts_bbox_head.pc_range)
        self.ego_cross_attn_static = OPUSCrossAttention(
            self.out_dim, 8, drop_out, self.pts_bbox_head.pc_range)
        self.ego_dynamic_proj = nn.Linear(self.out_dim, self.out_dim)
        self.ego_static_proj = nn.Linear(self.out_dim, self.out_dim)
        self.ego_fusion_norm = nn.LayerNorm(self.out_dim)
        # Role and ego routing have separate curricula.  Keep the historical
        # ``teacher_forcing`` key as a backwards-compatible fallback, but do
        # not let a pose curriculum silently control role labels (or vice
        # versa).
        legacy_tf = float(cfg.get('teacher_forcing', 0.0))
        self.prescf_role_teacher_forcing = float(
            cfg.get('role_teacher_forcing', legacy_tf))
        self.prescf_ego_teacher_forcing = float(
            cfg.get('ego_teacher_forcing', legacy_tf))
        self.prescf_teacher_forcing = self.prescf_ego_teacher_forcing
        self.prescf_loss_weights = dict(
            role=float(cfg.get('lambda_role', 0.1)),
            ego=float(cfg.get('lambda_ego', 0.1)),
            static=float(cfg.get('lambda_static', 0.01)),
            dynamic=float(cfg.get('lambda_dynamic', 0.01)),
            smooth=float(cfg.get('lambda_smooth', 0.01)),
            leak=float(cfg.get('lambda_leak', 0.01)))
        self.prescf_next_role_loss_weight = float(cfg.get(
            'next_role_loss_weight', 1.0))
        if self.prescf_next_role_loss_weight <= 0:
            raise ValueError('next_role_loss_weight must be positive')
        self.prescf_stage1_end_epoch = int(cfg.get('stage1_end_epoch', 5))
        self.prescf_stage2_end_epoch = int(cfg.get('stage2_end_epoch', 16))
        self.prescf_interaction_ramp_epochs = max(int(
            cfg.get('interaction_ramp_epochs', 4)), 1)
        self.prescf_joint_ramp_epochs = max(int(
            cfg.get('joint_ramp_epochs', 4)), 1)
        self.prescf_stage_gate_floor = float(cfg.get(
            'stage_gate_floor', 0.1))
        if not 0.0 < self.prescf_stage_gate_floor <= 1.0:
            raise ValueError('stage_gate_floor must be in (0, 1]')
        curriculum = cfg.get('forecast_curriculum', [1, 2, 3, 6])
        if not isinstance(curriculum, (list, tuple)) or not curriculum:
            raise TypeError('forecast_curriculum must be a non-empty sequence')
        self.prescf_forecast_curriculum = tuple(
            max(1, min(int(step), self.num_fu_frames)) for step in curriculum)
        self.prescf_stage = 1
        self.prescf_enable_interaction = False
        self.prescf_enable_joint_refine = False
        nn.init.zeros_(self.prescf_ego_pose_head[-1].weight)
        with torch.no_grad():
            self.prescf_ego_pose_head[-1].bias.copy_(
                torch.tensor([0.0, 0.0, 0.0, 1.0]))
        # Snapshot for reducing only rank-local TASS count increments.  It is
        # intentionally non-persistent and initialized after checkpoint load
        # on the first distributed forward.
        self._prescf_synced_tass_counts = None
        self._prescf_tass_snapshot = None
        self._configure_prescf_stage()

    def _configure_prescf_stage(self):
        """Freeze the perception/TASS anchor during the initial PreSCF stage."""
        cfg = self.dsqe_cfg
        self._prescf_tass_frozen = bool(cfg.get('freeze_tass', False))
        self._prescf_unfreeze_tass_epoch = int(
            cfg.get('stage3_start_epoch', 10 ** 9)) if cfg.get(
                'unfreeze_tass', True) else 10 ** 9
        self._prescf_unfreeze_tass_layers = int(
            cfg.get('unfreeze_tass_layers', 2))
        if cfg.get('freeze_backbone', False):
            for module in (getattr(self, 'img_backbone', None),
                           getattr(self, 'img_neck', None)):
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad = False
        if cfg.get('freeze_baseline_heads', True):
            # These modules belong to the established BaseLine current/SCF
            # and planning path.  PreSCF reads their representations but does
            # not alter their weights during Stages 1/2.
            baseline_modules = (
                self.plan_head, self.ego_cross_attn, self.position_encoder,
                self.reg_branch, self.vel_branch, self.cls_branch,
                self.points_scale_branch, self.traj_head)
            for module in baseline_modules:
                for parameter in module.parameters():
                    parameter.requires_grad = False
        if self._prescf_tass_frozen:
            # OPUSHead owns RAP/TASS and the initial Query bank.  Keeping it
            # fixed makes the BaseLine and PreSCF comparisons reproducible.
            for parameter in self.pts_bbox_head.parameters():
                parameter.requires_grad = False
            # DDP constructs its reducer once and does not support adding a
            # formerly-frozen parameter midway through training.  Keep only
            # the decoder layers scheduled for Stage 3 in the reducer and
            # optimizer from the start, while a gradient hook makes them an
            # exact zero-gradient anchor during Stages 1/2.  Their optimizer
            # group has zero weight decay, so a zero gradient cannot change
            # them through decoupled AdamW decay either.
            if cfg.get('unfreeze_tass', True) and \
                    self._prescf_unfreeze_tass_layers > 0:
                layers = getattr(getattr(
                    self.pts_bbox_head, 'transformer', None), 'decoder', None)
                layers = getattr(layers, 'decoder_layers', None)
                if layers is not None:
                    for layer in list(layers)[
                            -self._prescf_unfreeze_tass_layers:]:
                        for parameter in layer.parameters():
                            parameter.requires_grad = True
                            parameter.register_hook(
                                self._prescf_tass_gradient_gate)
            # ``loss_single_mask`` historically updates this buffer online;
            # freezing parameters alone would still change the TASS query
            # assignment every iteration.
            self.pts_bbox_head._prescf_freeze_tass_updates = True

    def _prescf_tass_gradient_gate(self, gradient):
        """Keep staged TASS parameters DDP-visible but frozen until Stage 3."""
        if self._prescf_tass_frozen:
            return torch.zeros_like(gradient)
        return gradient

    def _freeze_prescf_tass_state(self):
        """Capture and restore the non-parameter TASS assignment state.

        ``num_stamps_all`` is a buffer rather than a Parameter and therefore
        is not covered by the ordinary ``requires_grad`` freeze.  Keeping a
        clone of both the counts and the derived stamp indices makes the
        frozen BaseLine anchor deterministic in single- and multi-GPU runs.
        """
        head = self.pts_bbox_head
        if self._prescf_tass_snapshot is None:
            counts = getattr(head, 'num_stamps_all', None)
            indices = getattr(head, 'ind_stamps_all', None)
            if counts is not None:
                self._prescf_tass_snapshot = (
                    counts.detach().clone(),
                    None if indices is None else indices.detach().clone())
        elif self._prescf_tass_snapshot[1] is None and \
                getattr(head, 'ind_stamps_all', None) is not None:
            self._prescf_tass_snapshot = (
                self._prescf_tass_snapshot[0],
                head.ind_stamps_all.detach().clone())
        snapshot = self._prescf_tass_snapshot
        if snapshot is None:
            return
        counts, indices = snapshot
        head.num_stamps_all.copy_(counts.to(head.num_stamps_all))
        if indices is not None:
            head.ind_stamps_all = indices.to(
                device=head.num_stamps_all.device).clone()

    def _set_prescf_stage(self, epoch):
        if epoch < self.prescf_stage1_end_epoch:
            stage = 1
        elif epoch < self.prescf_stage2_end_epoch:
            stage = 2
        else:
            stage = 3
        self.prescf_stage = stage
        # All PreSCF blocks are part of the Stage-1 graph.  Curriculum is
        # expressed by their smooth positive gates below, never by bypassing
        # a module (which would produce unused parameters under DDP).
        self.prescf_enable_interaction = True
        self.prescf_enable_joint_refine = True

    def _prescf_stage_gates(self):
        """Return smooth module gates while retaining a complete DDP graph."""
        if not self.training:
            return 1.0, 1.0
        interaction = min(max(
            (self.curr_epoch - self.prescf_stage1_end_epoch + 1) /
            self.prescf_interaction_ramp_epochs, 0.0), 1.0)
        joint = min(max(
            (self.curr_epoch - self.prescf_stage2_end_epoch + 1) /
            self.prescf_joint_ramp_epochs, 0.0), 1.0)
        # Stage 1 still trains interaction and joint refinement.  A small
        # positive blend keeps the branch numerically close to its BaseLine
        # anchor while ensuring every expected module receives gradients.
        floor = self.prescf_stage_gate_floor if hasattr(
            self, 'prescf_stage_gate_floor') else 0.1
        return floor + (1.0 - floor) * interaction, floor + (1.0 - floor) * joint

    def init_weights(self):
        self.pts_bbox_head.init_weights()
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)
        # Warm-start the absolute PreSCF semantic head from the trained
        # BaseLine classifier.  Copy every layer of the classifier MLP; the
        # final projection retains an independent 17-way vector for each
        # refine point, so no point-specific weights are averaged.
        if self.dsqe_mode == 'prescf' and hasattr(self, 'joint_refine'):
            target = self.joint_refine.semantic_head
            with torch.no_grad():
                for index in (0, 2, 4):
                    source = self.cls_branch[index]
                    destination = target[index]
                    if (isinstance(source, nn.Linear) and
                            isinstance(destination, nn.Linear) and
                            source.weight.shape == destination.weight.shape):
                        destination.weight.copy_(source.weight)
                        destination.bias.copy_(source.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Migrate a BaseLine classifier into a missing PreSCF semantic head.

        ``init_weights`` runs before the epoch-56 checkpoint is loaded, so a
        one-time copy there would only copy random initialization.  Injecting
        the averaged point-specific BaseLine projection at state-dict load
        time makes warm-starting work for real BaseLine checkpoints while
        leaving native PreSCF checkpoints untouched.
        """
        if self.dsqe_mode == 'prescf' and hasattr(self, 'joint_refine'):
            target = self.joint_refine.semantic_head
            # A BaseLine checkpoint has ``cls_branch.{0,2,4}``; inject the
            # corresponding full MLP tensors only when a native PreSCF head
            # is absent.  This preserves strict loading for native PreSCF
            # checkpoints and makes the warm-start function-equivalent.
            for index in (0, 2, 4):
                source_weight = prefix + 'cls_branch.{}.weight'.format(index)
                source_bias = prefix + 'cls_branch.{}.bias'.format(index)
                target_weight = prefix + 'joint_refine.semantic_head.{}.weight'.format(index)
                target_bias = prefix + 'joint_refine.semantic_head.{}.bias'.format(index)
                if target_weight not in state_dict and source_weight in state_dict:
                    source = state_dict[source_weight]
                    if source.shape == target[index].weight.shape:
                        state_dict[target_weight] = source.clone()
                if target_bias not in state_dict and source_bias in state_dict:
                    source = state_dict[source_bias]
                    if source.shape == target[index].bias.shape:
                        state_dict[target_bias] = source.clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def set_epoch(self, epoch):
        self.curr_epoch = epoch
        if self.dsqe_mode == 'prescf':
            self._set_prescf_stage(epoch)
            start = int(self.dsqe_cfg.get('teacher_forcing_start_epoch',
                                          self.finetune_epoch))
            end = int(self.dsqe_cfg.get('teacher_forcing_end_epoch',
                                        start + 12))
            role_initial = float(self.dsqe_cfg.get(
                'role_teacher_forcing',
                self.dsqe_cfg.get('teacher_forcing', 0.0)))
            ego_initial = float(self.dsqe_cfg.get(
                'ego_teacher_forcing',
                self.dsqe_cfg.get('teacher_forcing', 0.0)))
            if epoch <= start:
                self.prescf_role_teacher_forcing = role_initial
                self.prescf_ego_teacher_forcing = ego_initial
            elif epoch >= end:
                self.prescf_role_teacher_forcing = 0.0
                self.prescf_ego_teacher_forcing = 0.0
            else:
                decay = 1.0 - (epoch - start) / max(end - start, 1)
                self.prescf_role_teacher_forcing = role_initial * decay
                self.prescf_ego_teacher_forcing = ego_initial * decay
            self.prescf_teacher_forcing = self.prescf_ego_teacher_forcing
            # Stage 3 optionally unfreezes only the final TASS/SCF decoder
            # layers, keeping the BaseLine anchor stable earlier in training.
            if self._prescf_tass_frozen and epoch >= self._prescf_unfreeze_tass_epoch:
                self._prescf_tass_frozen = False
                self.pts_bbox_head._prescf_freeze_tass_updates = False
        if self.dsqe_mode == 'prescf':
            # PreSCF is initialized from the finished epoch-56 BaseLine.  Its
            # epoch 0 is already a future-state fine-tuning stage; inheriting
            # the historical BaseLine ``pretrain`` flag would multiply every
            # future occupancy loss by zero and leave the absolute semantic
            # head without gradients.
            self.pretrain = False
            self.pts_bbox_head.pretrain = False
            if self._prescf_tass_frozen:
                self._freeze_prescf_tass_state()
            else:
                self._synchronize_tass_state(self)
                stamps = self.pts_bbox_head.num_stamps_all.float()
                stamps = stamps / stamps.sum(-1, keepdim=True).clamp_min(1e-6)
                self.pts_bbox_head.ind_stamps_all = get_matched_inds(
                    stamps, [self.num_query] + self.num_fu_query)
                self._synchronize_tass_state(self)
                self.pts_bbox_head.reset_mask()
        elif epoch<self.finetune_epoch:
            self.pretrain = True
            self.pts_bbox_head.pretrain = True
            if getattr(self.pts_bbox_head, 'num_stamps_all', None) is not None:
                self.pts_bbox_head.num_stamps_all[:] = 1  # avoid diving 0
        else:
            self.pretrain = False
            self.pts_bbox_head.pretrain = False
            num_stamps = self.pts_bbox_head.num_stamps_all / torch.sum(self.pts_bbox_head.num_stamps_all, dim=-1,
                                                                       keepdim=True)
            self.pts_bbox_head.ind_stamps_all = get_matched_inds(num_stamps, [self.num_query] + self.num_fu_query)

            self.pts_bbox_head.reset_mask()


    def trans_points(self, points_proposal, points_delta, trans_matrix):
        trans_matrix = trans_matrix.to(device=points_proposal.device,
                                       dtype=points_proposal.dtype)
        inv_trans_matrix = torch.linalg.inv(trans_matrix)
        points_proposal = decode_points(points_proposal, self.pc_range)
        # points_proposal = points_proposal.mean(dim=2, keepdim=True) fengze
        new_points = torch.matmul(points_proposal, trans_matrix[..., :3, :3].transpose(1, 2)) + trans_matrix[..., None,
                                                                                                :3, 3]
        new_points = new_points + points_delta
        new_points = torch.matmul(new_points, inv_trans_matrix[..., :3, :3].transpose(1, 2)) + inv_trans_matrix[...,
                                                                                               None, :3, 3]

        return encode_points(new_points, self.pc_range)

    def _num_forecast_frames(self):
        """Return the active forecast curriculum length."""
        if not self.training:
            return self.num_fu_frames
        if self.dsqe_mode == 'prescf':
            configured = self.dsqe_cfg.get('forecast_steps')
            if configured is not None:
                return max(1, min(int(configured), self.num_fu_frames))
            # Keep Stage 1 genuinely one-step.  Stage 2 follows the explicit
            # 1 -> 2 -> 3 -> 6 curriculum before retaining six steps in
            # Stage 3.
            stage1_end = int(self.dsqe_cfg.get(
                'stage1_end_epoch', getattr(self, 'prescf_stage1_end_epoch', 5)))
            if self.curr_epoch < stage1_end:
                return self.prescf_forecast_curriculum[0] if hasattr(
                    self, 'prescf_forecast_curriculum') else 1
            curriculum = getattr(self, 'prescf_forecast_curriculum',
                                 (1, 2, 3, 6))
            if len(curriculum) == 1:
                return curriculum[0]
            stage2_end = int(self.dsqe_cfg.get(
                'stage2_end_epoch', getattr(self, 'prescf_stage2_end_epoch',
                                            stage1_end + 1)))
            span = max(stage2_end - stage1_end, 1)
            progress = self.curr_epoch - stage1_end
            # Split Stage 2 into equal, deterministic bins.  The last bin
            # reaches the full configured horizon before Stage 3 begins.
            bin_index = min((progress * (len(curriculum) - 1)) // span,
                            len(curriculum) - 2)
            return curriculum[bin_index + 1]
        return max(1, min(self.curr_epoch - self.finetune_epoch + 1,
                          self.num_fu_frames))

    def refine_points(self, points_proposal, points_delta):
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, self.num_refines, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    @staticmethod
    def _to_batch_tensor(value, device, dtype):
        if hasattr(value, 'stack') and hasattr(value, 'data'):
            value = value.data
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, (list, tuple)):
            return torch.as_tensor(value, device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _to_batch_sequence(value, device, dtype):
        """Convert a collated or raw ragged field to per-sample tensors.

        ``DataContainer(stack=False)`` is collated by MMCV as
        ``[[sample_0, sample_1, ...]]`` for each worker micro-batch.  Actor
        rows must remain ragged; blindly passing that nested list to
        ``torch.as_tensor`` raises for batch sizes greater than one and, more
        importantly, would destroy the actor/sample correspondence.
        """
        if hasattr(value, 'stack') and hasattr(value, 'data'):
            value = value.data

        # Unwrap the outer micro-batch container emitted by mmcv.collate.
        if isinstance(value, (list, tuple)) and len(value) == 1 and \
                isinstance(value[0], (list, tuple)):
            value = value[0]

        def convert(item):
            if hasattr(item, 'stack') and hasattr(item, 'data'):
                item = item.data
            if torch.is_tensor(item):
                tensor = item.to(device=device, dtype=dtype)
            else:
                tensor = torch.as_tensor(np.asarray(item), device=device,
                                         dtype=dtype)
            # A singleton batch axis can survive a one-sample DataContainer;
            # remove only that axis, never actor rows.
            while tensor.ndim > 2 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            return tensor

        if value is None:
            return None
        if torch.is_tensor(value):
            if value.ndim <= 2:
                return [convert(value)]
            return [convert(value[index]) for index in range(value.shape[0])]
        if isinstance(value, (list, tuple)):
            if not value:
                return []
            # After unwrapping ``DataContainer.data``, a collated ragged
            # batch is a list of tensors.  This includes one-dimensional
            # label tensors, which must not be mistaken for one unbatched
            # Python list of scalar labels.
            if all(torch.is_tensor(item) or (
                    hasattr(item, 'stack') and hasattr(item, 'data'))
                   for item in value):
                return [convert(item) for item in value]
            # A raw unbatched list of scalar/row values is one sample.  A
            # collated batch contains matrix-like elements and is split by
            # sample instead.
            def ndim(item):
                if torch.is_tensor(item):
                    return item.ndim
                try:
                    return np.asarray(item, dtype=object).ndim
                except Exception:
                    return 0
            if all(ndim(item) <= 1 for item in value):
                return [convert(value)]
            return [convert(item) for item in value]
        tensor = SparseWorld4DTraj._to_batch_tensor(value, device, dtype)
        return [convert(tensor[index]) for index in range(tensor.shape[0])] \
            if tensor.ndim >= 3 else [convert(tensor)]

    @staticmethod
    def _synchronize_tass_state(model):
        import torch.distributed as dist
        head = model.pts_bbox_head
        if not dist.is_available() or not dist.is_initialized():
            return
        previous = getattr(model, '_prescf_synced_tass_counts', None)
        if previous is None or previous.shape != head.num_stamps_all.shape or \
                previous.device != head.num_stamps_all.device:
            # This also handles checkpoint load: the persisted history is
            # broadcast once, rather than summed once per rank.
            dist.broadcast(head.num_stamps_all, src=0)
        else:
            delta = (head.num_stamps_all - previous).to(torch.float64)
            dist.all_reduce(delta, op=dist.ReduceOp.SUM)
            synchronized = previous.to(delta.dtype) + delta
            head.num_stamps_all.copy_(
                synchronized.round().to(head.num_stamps_all.dtype))
        model._prescf_synced_tass_counts = head.num_stamps_all.clone()
        if getattr(head, 'ind_stamps_all', None) is not None:
            dist.broadcast(head.ind_stamps_all, src=0)

    def _ensure_tass_state(self):
        head = self.pts_bbox_head
        if getattr(self, '_prescf_tass_frozen', False):
            self._freeze_prescf_tass_state()
        if getattr(head, 'ind_stamps_all', None) is None:
            stamps = head.num_stamps_all.float()
            stamps = stamps / stamps.sum(-1, keepdim=True).clamp_min(1e-6)
            head.ind_stamps_all = get_matched_inds(
                stamps, [self.num_query] + self.num_fu_query)
            if hasattr(head, 'reset_mask'):
                head.reset_mask()
        if getattr(self, 'dsqe_enabled', False):
            self._synchronize_tass_state(self)
            if getattr(self, '_prescf_tass_frozen', False):
                self._freeze_prescf_tass_state()

    def _build_role_metadata(self, kwargs, interval, gt_relative_matrices=None):
        """Build actor role targets in the *future ego frame*.

        The temporal nuScenes/VAD cache stores ``gt_agent_fut_trajs`` as
        adjacent displacements in the current LiDAR frame.  This method first
        converts the actor box and every displacement vector to the E0 ego
        frame, cumulatively integrates the requested horizon, and only then
        applies the E0->Et ego-frame transform.  All returned centers, box
        dimensions and yaw values therefore share the occupancy/Query frame.
        """
        boxes = kwargs.get('temporal_agent_boxes')
        feats = kwargs.get('temporal_agent_feats')
        labels = kwargs.get('temporal_agent_labels')
        lidar2ego = kwargs.get('temporal_agent_lidar2ego')
        if boxes is None:
            return None
        if hasattr(boxes, 'stack') and hasattr(boxes, 'data'):
            boxes = boxes.data
        if hasattr(feats, 'stack') and hasattr(feats, 'data'):
            feats = feats.data
        if hasattr(labels, 'stack') and hasattr(labels, 'data'):
            labels = labels.data
        def nested_tensor_device(value):
            if torch.is_tensor(value):
                return value.device
            if hasattr(value, 'stack') and hasattr(value, 'data'):
                return nested_tensor_device(value.data)
            if isinstance(value, (list, tuple)):
                for item in value:
                    found = nested_tensor_device(item)
                    if found is not None:
                        return found
            return None

        device = (gt_relative_matrices.device if torch.is_tensor(
            gt_relative_matrices) else nested_tensor_device(boxes))
        if device is None:
            device = torch.device('cpu')
        box_list = self._to_batch_sequence(boxes, device, torch.float32)
        feat_list = None if feats is None else self._to_batch_sequence(
            feats, device, torch.float32)
        label_list = None if labels is None else self._to_batch_sequence(
            labels, device, torch.long)
        l2e_list = None if lidar2ego is None else self._to_batch_sequence(
            lidar2ego, device, torch.float32)
        if (torch.is_tensor(labels) and labels.ndim >= 2 and
                labels.shape[0] == len(box_list) and len(box_list) > 1):
            label_list = [labels[index].to(device=device, dtype=torch.long)
                          for index in range(labels.shape[0])]
        elif (torch.is_tensor(labels) and labels.ndim == 2 and
              len(box_list) == 1 and labels.shape[0] == 1):
            label_list = [labels[0].to(device=device, dtype=torch.long)]
        output = []
        dt = float(self.dsqe_cfg.get('role_frame_dt', 0.5))
        inflation = float(self.dsqe_cfg.get('role_box_inflation', 0.5))
        trajectory_mode = str(self.dsqe_cfg.get(
            'role_trajectory_mode', 'increment')).lower()
        if trajectory_mode not in ('absolute', 'increment', 'increments'):
            raise ValueError(
                'role_trajectory_mode must be absolute or increment, got {}'.format(
                    trajectory_mode))
        dynamic_ids = set(int(x) for x in self.dynamic_class_ids)
        adjacent = self._build_adjacent_ego_targets(
            kwargs, 1 if not box_list else len(box_list), interval,
            device, torch.float32)
        cumulative = self.ego_warp.identity(
            len(box_list), device, torch.float32)
        for step in range(min(interval, len(adjacent))):
            cumulative = self.ego_warp.compose(cumulative, adjacent[step])
        to_future = self.ego_warp.inverse(cumulative)
        for b, current in enumerate(box_list):
            if current.ndim == 1:
                current = current.unsqueeze(0)
            centers = current[:, :3]
            num_agents = centers.shape[0]
            if l2e_list is not None and b < len(l2e_list):
                lidar_to_ego = l2e_list[b]
                if lidar_to_ego.ndim == 3:
                    lidar_to_ego = lidar_to_ego[0]
            else:
                lidar_to_ego = torch.eye(4, device=device, dtype=torch.float32)
            lidar_rot = lidar_to_ego[:3, :3]
            centers_e0 = torch.matmul(
                centers, lidar_rot.transpose(0, 1)) + lidar_to_ego[:3, 3]
            lidar_yaw = torch.atan2(lidar_rot[1, 0], lidar_rot[0, 0])
            # The stored actor future trajectory is indexed by adjacent
            # horizon and remains in the source LiDAR frame until the vector
            # rotation below.  ``absolute`` is retained only for old private
            # caches whose preprocessing already integrated the trajectory.
            displacement = current.new_zeros(num_agents, 2)
            yaw_delta = current.new_zeros(num_agents)
            future_valid = torch.ones(num_agents, dtype=torch.bool,
                                       device=device)
            if feat_list is not None and b < len(feat_list):
                agent_feat = feat_list[b]
                if agent_feat.ndim == 1:
                    agent_feat = agent_feat.unsqueeze(0)
                dims = 2 * self.num_fu_frames
                if agent_feat.shape[-1] >= dims:
                    trajectory = agent_feat[:, :dims].reshape(
                        -1, self.num_fu_frames, 2)
                    step = max(0, min(interval - 1,
                                      self.num_fu_frames - 1))
                    n = min(num_agents, trajectory.shape[0])
                    if n:
                        if trajectory_mode == 'absolute':
                            displacement[:n] = trajectory[:n, step]
                        else:
                            displacement[:n] = trajectory[:n, :step + 1].sum(1)
                    mask_start = dims
                    mask_end = mask_start + self.num_fu_frames
                    if interval > 0 and agent_feat.shape[-1] >= mask_end:
                        mask_values = agent_feat[
                            :n, mask_start:mask_start + self.num_fu_frames]
                        # Validity is defined at the requested horizon.  An
                        # actor that is absent at an earlier horizon cannot be
                        # revived by a later one in standard nuScenes data,
                        # but selecting the target slot is the correct
                        # interpretation for custom ragged caches as well.
                        future_valid.zero_()
                        future_valid[:n] = mask_values[:, step] > 0.5
                    yaw_start = agent_feat.shape[-1] - self.num_fu_frames
                    if interval > 0 and yaw_start >= mask_end and n:
                        yaw_values = agent_feat[
                            :n, yaw_start:yaw_start + self.num_fu_frames]
                        # VAD stores one scalar raw adjacent yaw delta per
                        # future frame, not a sin/cos pair.  The optional
                        # sin/cos mode is retained for private legacy caches.
                        if str(self.dsqe_cfg.get(
                                'role_yaw_encoding', 'raw')).lower() in (
                                    'sincos', 'sin_cos'):
                            yaw_values = torch.atan2(
                                yaw_values.sin(), yaw_values.cos())
                        if trajectory_mode == 'absolute':
                            yaw_delta[:n] = yaw_values[:, step]
                        else:
                            yaw_delta[:n] = yaw_values[:, :step + 1].sum(1)
            # Trajectory vectors are free vectors, so only the LiDAR->ego
            # rotation applies (not the sensor translation).
            displacement_e0 = torch.matmul(
                torch.cat([displacement,
                           displacement.new_zeros(num_agents, 1)], -1),
                lidar_rot.transpose(0, 1))[:, :2]
            future_centers_t0 = centers_e0.clone()
            if interval > 0 and displacement.shape[0] == num_agents:
                future_centers_t0[:, :2] += displacement_e0
            future_centers = self.ego_warp.transform_metric(
                future_centers_t0.unsqueeze(0).unsqueeze(2),
                to_future[b:b + 1])[0, :, 0]
            transform_yaw = torch.atan2(to_future[b, 1, 0],
                                        to_future[b, 0, 0])
            current_yaw = (current[:, 6] if current.shape[-1] >= 7 else
                           centers.new_zeros(num_agents))
            # ``gt_boxes`` uses SECOND yaw: second = -raw - pi/2.  The
            # current box is in LiDAR, while ``to_future`` maps E0 -> Et.
            # Thus both frame rotations and VAD's raw actor delta subtract in
            # SECOND space.
            future_yaw = current_yaw - lidar_yaw - yaw_delta - transform_yaw
            future_yaw = (future_yaw + torch.pi) % (2 * torch.pi) - torch.pi
            if current.shape[-1] >= 6:
                # The VAD converter concatenates ``Box.wlh`` and converts
                # orientation to SECOND yaw.  Under that yaw convention the
                # local x/y half extents pair with (width,length), so preserve
                # the native order.  Normalize alternate private layouts to
                # the same SECOND-aligned (width,length,height) contract.
                raw_dims = current[:, 3:6].abs()
                dims_order = str(self.dsqe_cfg.get(
                    'role_box_dims_order', 'wlh')).lower()
                if dims_order == 'wlh':
                    dims = raw_dims
                elif dims_order == 'lwh':
                    dims = raw_dims[:, [1, 0, 2]]
                elif dims_order == 'lhw':
                    dims = raw_dims[:, [2, 0, 1]]
                else:
                    raise ValueError(
                        'role_box_dims_order must be lwh/lhw or wlh, got {}'.format(
                            dims_order))
                radius = 0.5 * dims[:, :2].square().sum(-1).sqrt() + inflation
            else:
                dims = centers.new_ones(num_agents, 3)
                radius = centers.new_full((num_agents,), 1.0 + inflation)
            actor_labels = None
            if label_list is not None and b < len(label_list):
                actor_labels = label_list[b].reshape(-1)[:num_agents]
                if actor_labels.numel() != num_agents:
                    actor_labels = None
            if actor_labels is not None:
                known = actor_labels >= 0
                actor_dynamic = torch.zeros(
                    num_agents, device=device, dtype=torch.bool)
                for class_id in dynamic_ids:
                    actor_dynamic |= actor_labels == class_id
                actor_role = actor_dynamic.to(torch.float32)
                valid = future_valid & known
            else:
                # Fallback for old caches without labels: velocity is only a
                # weak prior, never a class-0/``others`` dynamic label.
                speed = (displacement.norm(-1) / max(interval * dt, 1e-3)
                         if interval > 0 else displacement.new_zeros(num_agents))
                actor_role = (speed > float(self.dsqe_cfg.get(
                    'role_speed_threshold', 0.5))).float()
                valid = future_valid
            output.append(dict(
                centers=future_centers, radius=radius.clamp_min(0.5),
                role=actor_role, valid=valid,
                labels=(actor_labels if actor_labels is not None else
                        torch.full((num_agents,), -1, device=device,
                                   dtype=torch.long)),
                dims=dims, yaw=future_yaw, inflation=inflation)
            )
        return output

    @staticmethod
    def _role_targets_from_metadata(points_metric, metadata):
        """Match query points to actor footprints for GT role forcing."""
        bsz, queries, num_points = points_metric.shape[:3]
        target = points_metric.new_zeros(bsz, queries, num_points, 1)
        valid = torch.zeros(bsz, queries, num_points, 1,
                            device=points_metric.device, dtype=torch.bool)
        if metadata is None:
            return target, valid
        for b in range(min(bsz, len(metadata))):
            actors = metadata[b]
            if not actors or actors.get('centers') is None:
                continue
            centers = actors['centers'].to(points_metric)
            if centers.numel() == 0:
                continue
            radius = actors['radius'].to(points_metric)
            role = actors['role'].to(points_metric)
            actor_valid = actors.get('valid', torch.ones_like(role, dtype=torch.bool))
            actor_valid = actor_valid.to(points_metric.device).bool()
            dims = actors.get('dims')
            yaw = actors.get('yaw')
            inflation = torch.as_tensor(
                actors.get('inflation', 0.5), device=points_metric.device,
                dtype=points_metric.dtype)
            points = points_metric[b].reshape(-1, 3)
            distance = torch.cdist(points[:, :2], centers[:, :2])
            nearest, index = distance.min(dim=1)
            matched = (nearest <= radius[index]) & actor_valid[index]
            if dims is not None and yaw is not None:
                dims = dims.to(points_metric).abs()
                yaw = yaw.to(points_metric)
                delta = points[:, :2] - centers[index, :2]
                c, s = yaw[index].cos(), yaw[index].sin()
                local_x = c * delta[:, 0] + s * delta[:, 1]
                local_y = -s * delta[:, 0] + c * delta[:, 1]
                matched &= (local_x.abs() <= dims[index, 0] * 0.5 + inflation)
                matched &= (local_y.abs() <= dims[index, 1] * 0.5 + inflation)
            target[b].reshape(-1, 1)[matched] = role[index[matched], None]
            valid[b].reshape(-1, 1)[matched] = True
        return target, valid

    @staticmethod
    def _query_actor_ids(point_actor_id, actor_valid=None):
        """Reduce point associations to a stable per-Query actor ID.

        IDs are batch-local row indices in ``temporal_agent_*``.  Majority
        voting only considers valid dynamic point associations; static map
        classes and unmatched points retain ``-1``.
        """
        batch, queries = point_actor_id.shape[:2]
        query_actor_id = point_actor_id.new_full((batch, queries), -1)
        if actor_valid is None:
            actor_valid = point_actor_id >= 0
        for b in range(batch):
            for q in range(queries):
                ids = point_actor_id[b, q][actor_valid[b, q] &
                                                (point_actor_id[b, q] >= 0)]
                if ids.numel():
                    values, counts = torch.unique(ids, return_counts=True)
                    query_actor_id[b, q] = values[counts.argmax()]
        return query_actor_id

    @classmethod
    def _initialize_actor_association(
            cls, points_metric, metadata, max_distance=None):
        """Associate newly activated Query points with GT dynamic actors.

        The association is created once from the actor's inflated footprint
        and then carried recursively.  It never depends on predicted role
        probability and is not recomputed when a carried Query drifts.
        """
        batch, queries, num_points = points_metric.shape[:3]
        point_actor_id = torch.full(
            (batch, queries, num_points), -1, device=points_metric.device,
            dtype=torch.long)
        actor_valid = torch.zeros_like(point_actor_id, dtype=torch.bool)
        if metadata is not None:
            for b in range(min(batch, len(metadata))):
                actors = metadata[b]
                if not actors or actors.get('centers') is None:
                    continue
                centers = actors['centers'].to(points_metric)
                if not centers.numel():
                    continue
                role = actors['role'].to(points_metric) > 0.5
                valid = actors.get(
                    'valid', torch.ones_like(role, dtype=torch.bool))
                valid = valid.to(points_metric.device).bool() & role
                radius = actors['radius'].to(points_metric)
                points = points_metric[b].reshape(-1, 3)
                distance = torch.cdist(points[:, :2], centers[:, :2])
                nearest, index = distance.min(dim=1)
                matched = (nearest <= radius[index]) & valid[index]
                if max_distance is not None:
                    matched &= nearest <= float(max_distance)
                dims, yaw = actors.get('dims'), actors.get('yaw')
                if dims is not None and yaw is not None:
                    dims = dims.to(points_metric).abs()
                    yaw = yaw.to(points_metric)
                    inflation = torch.as_tensor(
                        actors.get('inflation', 0.5),
                        device=points_metric.device, dtype=points_metric.dtype)
                    delta = points[:, :2] - centers[index, :2]
                    c, s = yaw[index].cos(), yaw[index].sin()
                    local_x = c * delta[:, 0] + s * delta[:, 1]
                    local_y = -s * delta[:, 0] + c * delta[:, 1]
                    matched &= (
                        local_x.abs() <= dims[index, 0] * 0.5 + inflation)
                    matched &= (
                        local_y.abs() <= dims[index, 1] * 0.5 + inflation)
                flat_ids = point_actor_id[b].reshape(-1)
                flat_valid = actor_valid[b].reshape(-1)
                flat_ids[matched] = index[matched]
                flat_valid[matched] = True
        return dict(
            point_actor_id=point_actor_id,
            query_actor_id=cls._query_actor_ids(point_actor_id),
            actor_valid=actor_valid)

    @classmethod
    def _recover_actor_association(
            cls, point_actor_id, points_metric, metadata, max_distance):
        """Persist newly reliable actor matches without rewriting identity.

        Reverse dynamic coverage may bring a previously-unmatched point into
        a valid actor footprint.  Such a point can acquire an association at
        its first reliable recovery, while every existing ID remains
        immutable even if later geometry drifts or another actor is nearer.
        """
        if tuple(point_actor_id.shape) != tuple(points_metric.shape[:3]):
            raise ValueError(
                'actor association shape {} does not match points {}'.format(
                    tuple(point_actor_id.shape),
                    tuple(points_metric.shape[:3])))
        if float(max_distance) <= 0:
            raise ValueError('actor recovery max distance must be positive')
        candidate = cls._initialize_actor_association(
            points_metric, metadata, max_distance=max_distance)
        recovered_mask = ((point_actor_id < 0) &
                          candidate['actor_valid'])
        updated = torch.where(
            recovered_mask, candidate['point_actor_id'], point_actor_id)
        actor_valid = cls._refresh_actor_association(updated, metadata)
        return dict(
            point_actor_id=updated,
            query_actor_id=cls._query_actor_ids(updated),
            actor_valid=actor_valid,
            recovered_mask=recovered_mask)

    @staticmethod
    def _refresh_actor_association(point_actor_id, metadata):
        """Apply actor future validity without changing persistent IDs."""
        actor_valid = torch.zeros_like(point_actor_id, dtype=torch.bool)
        if metadata is None:
            return actor_valid
        for b in range(min(point_actor_id.shape[0], len(metadata))):
            actors = metadata[b]
            if not actors or actors.get('role') is None:
                continue
            role = actors['role'].to(point_actor_id.device) > 0.5
            valid = actors.get(
                'valid', torch.ones_like(role, dtype=torch.bool))
            valid = valid.to(point_actor_id.device).bool() & role
            ids = point_actor_id[b]
            in_range = (ids >= 0) & (ids < valid.numel())
            actor_valid[b][in_range] = valid[ids[in_range]]
        return actor_valid

    @classmethod
    def _apply_actor_association_to_role_cache(
            cls, cache, point_actor_id, metadata):
        """Override associated point roles using persistent GT actor IDs."""
        if cache is None:
            return None
        cache = dict(cache)
        target = cache['role_target'].clone()
        valid = cache['role_valid'].clone()
        if target.shape != point_actor_id.shape:
            raise ValueError(
                'role cache shape {} does not match actor association {}'.format(
                    tuple(target.shape), tuple(point_actor_id.shape)))
        actor_valid = cls._refresh_actor_association(
            point_actor_id, metadata)
        associated = point_actor_id >= 0
        # Associations are created only for dynamic actors.  An invalid future
        # actor is ignored instead of being silently rematched as static.
        target[associated] = 1.0
        valid[associated] = actor_valid[associated]
        cache.update(
            role_target=target,
            role_valid=valid,
            point_actor_id=point_actor_id.clone(),
            query_actor_id=cls._query_actor_ids(point_actor_id),
            actor_valid=actor_valid)
        return cache

    @staticmethod
    def _prescf_supervision_caches(state):
        """Return the time-aligned role and future-geometry caches."""
        return (state.get('current_role_cache'),
                state.get('future_match_cache'))

    @staticmethod
    def _mask_new_query_role_cache(cache, num_carried):
        """Exclude stamp ``t+1`` Queries from the current-time role GT."""
        if cache is None or cache.get('role_target') is None:
            return cache
        cache = dict(cache)
        target = cache['role_target'].clone()
        valid = cache['role_valid'].clone()
        if num_carried < target.shape[1]:
            target[:, num_carried:] = 0
            valid[:, num_carried:] = False
        cache.update(role_target=target, role_valid=valid)
        return cache

    @staticmethod
    def _prescf_role_supervision_outputs(state):
        """Expose pure current predictions and next-state roles separately."""
        prediction = dict(
            role_logits=state['role_logits'],
            role_pred=state['role_pred'],
            query_role=state['pred_query_role'])
        next_state = dict(
            role_logits=state['next_role_logits'],
            role_pred=state['next_role_pred'],
            query_role=state['next_query_role'])
        return prediction, next_state

    def loss_traj(self, pred_traj, gt_traj, ego_interval):
        loss_dict = dict()
        loss_dict[f'loss_traj_{str(ego_interval)}s'] = self.l2_loss(pred_traj, gt_traj)

        return loss_dict

    def _baseline_foreground_mask(self, points, interval, img_metas, kwargs):
        """Preserve the reference foreground mask used by BaseLine loss."""
        if 'temporal_trajs' not in kwargs:
            return points[..., 0] >= 0
        matrices = points.new_tensor(np.stack(
            [meta['ego2lidar'] for meta in img_metas]))
        gt_traj = kwargs['temporal_trajs'][:, interval:interval + 1].to(
            device=points.device, dtype=points.dtype)
        offset = torch.cat([
            -gt_traj, torch.zeros_like(gt_traj[..., :1])], dim=-1)
        gt_points = self.trans_points(
            points.flatten(1, 2), offset, matrices).reshape_as(points)
        return gt_points[..., 0] >= 0

    def _prescf_foreground_mask(self, points, interval=None, img_metas=None,
                                kwargs=None):
        """Return a mask in the already-aligned PreSCF future frame.

        ``_baseline_foreground_mask`` intentionally applies the historical
        BaseLine ego-trajectory conversion to its input points.  PreSCF
        points, however, have already been warped into ``E_{t+1}`` by the
        recursive state transition.  Applying that conversion a second time
        would mix frames and make the occupancy supervision mask depend on
        the old residual-path convention.  The complete encoded xyz range is
        checked in this already-aligned frame for every future step.
        """
        return ((points >= 0) & (points < 1)).all(dim=-1)

    @staticmethod
    def _matrix_value(value, device, dtype, batch_size):
        """Normalize one collated transform value to ``[B, 4, 4]``."""
        if hasattr(value, 'stack') and hasattr(value, 'data'):
            value = value.data
        if isinstance(value, (list, tuple)) and value and all(
                torch.is_tensor(item) for item in value):
            value = torch.stack(list(value), dim=0)
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.ndim == 2:
            value = value.unsqueeze(0).expand(batch_size, -1, -1)
        elif value.ndim == 3 and value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1, -1)
        if value.ndim != 3 or value.shape[-2:] != (4, 4):
            raise ValueError('Expected ego transforms with shape [B,4,4], got {}'.format(
                tuple(value.shape)))
        if value.shape[0] != batch_size:
            raise ValueError('Ego transform batch {} does not match batch {}'.format(
                value.shape[0], batch_size))
        return value

    @classmethod
    def _sequence_values(cls, sequence, num_steps, device, dtype, batch_size):
        """Read collated dict/tensor transform sequences in time order."""
        values = []
        if sequence is None:
            return values
        if hasattr(sequence, 'stack') and hasattr(sequence, 'data'):
            sequence = sequence.data
        # ``stack=False`` DataContainers may arrive as one dict per sample.
        # Reassemble each time slot along the batch dimension before parsing.
        if isinstance(sequence, (list, tuple)) and sequence and all(
                isinstance(item, dict) for item in sequence):
            for index in range(num_steps):
                gathered = []
                for item in sequence:
                    value = item.get(index, item.get(str(index)))
                    if value is None:
                        gathered = []
                        break
                    gathered.append(value)
                if not gathered:
                    break
                values.append(cls._matrix_value(gathered, device, dtype,
                                                batch_size))
            return values
        if isinstance(sequence, dict):
            for index in range(num_steps):
                value = sequence.get(index, sequence.get(str(index)))
                if value is None:
                    break
                values.append(cls._matrix_value(value, device, dtype, batch_size))
        else:
            tensor = torch.as_tensor(sequence, device=device, dtype=dtype)
            if tensor.ndim == 4:
                values = [cls._matrix_value(tensor[:, index], device, dtype,
                                            batch_size)
                          for index in range(min(num_steps, tensor.shape[1]))]
            elif tensor.ndim == 3:
                values = [cls._matrix_value(tensor, device, dtype, batch_size)]
        return values

    @classmethod
    def _build_adjacent_ego_targets(cls, kwargs, batch_size, num_steps,
                                    device, dtype):
        """Build ``T(E_{t+1}->E_t)`` from either explicit or cumulative data.

        Dataset ``temporal2ego`` values are cumulative ``T(E_k->E_0)``.  The
        explicit ``temporal_adjacent2ego`` field is preferred, while the
        fallback computes the mathematically equivalent adjacent sequence.
        """
        explicit = cls._sequence_values(
            kwargs.get('temporal_adjacent2ego'), num_steps, device, dtype,
            batch_size)
        if len(explicit) >= num_steps:
            return explicit[:num_steps]
        cumulative = cls._sequence_values(
            kwargs.get('temporal2ego'), num_steps, device, dtype, batch_size)
        if not cumulative:
            return []
        identity = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(
            batch_size, -1, -1)
        adjacent = []
        previous = identity
        for current in cumulative[:num_steps]:
            adjacent.append(torch.matmul(torch.linalg.inv(previous), current))
            previous = current
        return adjacent

    @classmethod
    def _future_pose_target(cls, kwargs, interval, device, dtype,
                            batch_size=None):
        """Read one adjacent ego transform, never a cumulative transform."""
        if batch_size is None:
            batch_size = 1
        targets = cls._build_adjacent_ego_targets(
            kwargs, batch_size, interval + 1, device, dtype)
        return targets[interval] if len(targets) > interval else None

    def _forward_baseline_scf(self, outs, ego_feat, img_metas, kwargs):
        """The unmodified SparseWorld future SCF reference path."""
        stamps = self.pts_bbox_head.ind_stamps_all
        query_feat = outs['query_feat']
        query_pos = outs['all_refine_pts'][-1]
        query_cls = outs['all_cls_scores'][-1]
        batch_size = query_feat.shape[0]
        current_mask = stamps == 0
        current_feat = query_feat[:, current_mask]
        current_pos = query_pos[:, current_mask].detach()
        timestamp = current_pos.new_zeros(
            batch_size, self.num_query, self.num_refines, 1)
        forecast_points_list, forecast_semantics_list = [], []
        pred_trajs_list, forecast_points_mask_list = [], []
        num_forecast = (self._num_forecast_frames()
                        if hasattr(self, '_num_forecast_frames') else
                        max(1, min(self.curr_epoch - self.finetune_epoch + 1,
                                   self.num_fu_frames)))
        for interval in range(num_forecast):
            fused_ego_feat, _ = self.ego_cross_attn(
                ego_feat.new_full((batch_size, 1, 3), 0.5), ego_feat,
                current_pos.detach(), current_feat.detach())
            pred_trajs_list.append(self.traj_head(fused_ego_feat))
            new_mask = stamps == interval + 1
            current_feat = torch.cat(
                [current_feat, query_feat[:, new_mask]], dim=1)
            current_pos = torch.cat(
                [current_pos, query_pos[:, new_mask]], dim=1).detach()
            new_timestamp = current_pos.new_full(
                (batch_size, int(new_mask.sum()), self.num_refines, 1), 0.5)
            timestamp = torch.cat([timestamp, new_timestamp], dim=1)
            pos_embedding = self.position_encoder(
                torch.cat([current_pos, timestamp], dim=-1).flatten(2, 3))
            current_feat = current_feat + fused_ego_feat + pos_embedding
            reg_offset = self.reg_branch(current_feat).reshape(
                batch_size, -1, self.num_refines, 3) * 0.5
            semantics = self.cls_branch(current_feat).reshape(
                batch_size, -1, self.num_refines, 17)
            velocity = self.vel_branch(current_feat).reshape(
                batch_size, -1, self.num_refines, 2)
            labels = semantics.argmax(-1)
            moving = ((labels >= 2) & (labels <= 10)).unsqueeze(-1)
            reg_offset[..., :2] += velocity * moving
            current_pos = self.refine_points(
                current_pos, reg_offset.flatten(2, 3))
            forecast_semantics_list.append(semantics)
            forecast_points_list.append(current_pos)
            if self.training:
                forecast_points_mask_list.append(
                    self._baseline_foreground_mask(
                        current_pos, interval, img_metas, kwargs))
        if (not self.pretrain and
                len(pred_trajs_list) < self.num_fu_frames):
            fused_ego_feat, _ = self.ego_cross_attn(
                ego_feat.new_zeros(batch_size, 1, 3), ego_feat,
                current_pos, current_feat)
            pred_trajs_list.append(self.traj_head(fused_ego_feat))
        return dict(cls_score=query_cls[:, current_mask],
                    refine_pts=query_pos[:, current_mask], outs=outs,
                    forecast_semantics_list=forecast_semantics_list,
                    forecast_points_list=forecast_points_list,
                    pred_trajs_list=pred_trajs_list,
                    forecast_points_mask_list=forecast_points_mask_list,
                    prescf_outputs=None)

    def _forward_prescf(self, outs, ego_feat, img_metas, kwargs):
        """Direct Query-state recursion replacing the original SCF future step."""
        stamps = self.pts_bbox_head.ind_stamps_all
        all_feat = outs['query_feat']
        all_points = outs['all_refine_pts'][-1]
        all_semantics = outs['all_cls_scores'][-1]
        batch_size = all_feat.shape[0]
        current_mask = stamps == 0
        state_feat = all_feat[:, current_mask]
        state_points = all_points[:, current_mask]
        state_semantics = all_semantics[:, current_mask]
        state_role = None
        state_point_actor_id = None
        activation = state_points.new_zeros(
            batch_size, state_points.shape[1], dtype=torch.long)
        cumulative = self.ego_warp.identity(
            batch_size, state_points.device, state_points.dtype)
        predicted_cumulative = self.ego_warp.identity(
            batch_size, state_points.device, state_points.dtype)
        lidar_to_ego_value = kwargs.get('temporal_agent_lidar2ego')
        if lidar_to_ego_value is None and img_metas:
            fallback = []
            for meta in img_metas:
                ego_to_lidar = (meta.get('ego2lidar')
                                if isinstance(meta, dict) else None)
                if ego_to_lidar is None:
                    fallback = []
                    break
                fallback.append(torch.linalg.inv(torch.as_tensor(
                    ego_to_lidar, device=state_points.device,
                    dtype=state_points.dtype)))
            if fallback:
                lidar_to_ego_value = torch.stack(fallback)
        lidar_to_ego = (self._matrix_value(
            lidar_to_ego_value, state_points.device, state_points.dtype,
            batch_size) if lidar_to_ego_value is not None else
            self.ego_warp.identity(
                batch_size, state_points.device, state_points.dtype))
        previous_lidar_position = state_points.new_zeros(batch_size, 3)
        forecast_points, forecast_semantics = [], []
        pred_trajs, forecast_masks, prescf_outputs, match_cache_list = [], [], [], []
        num_forecast = (self._num_forecast_frames()
                        if hasattr(self, '_num_forecast_frames') else
                        self.num_fu_frames)
        adjacent_targets = (self._build_adjacent_ego_targets(
            kwargs, batch_size, num_forecast, state_points.device,
            state_points.dtype) if self.training else [])
        previous_motion = None
        previous_motion_valid = None
        role_metadata_cache = {}
        interaction_stage_gate, joint_stage_gate = self._prescf_stage_gates()

        def role_metadata_at(horizon):
            if horizon not in role_metadata_cache:
                role_metadata_cache[horizon] = self._build_role_metadata(
                    kwargs, horizon)
            return role_metadata_cache[horizon]

        for interval in range(num_forecast):
            new_mask = stamps == interval + 1
            num_carried = state_feat.shape[1]
            new_feat = all_feat[:, new_mask]
            new_points = all_points[:, new_mask]
            new_semantics = all_semantics[:, new_mask]
            num_new = new_feat.shape[1]
            # New RAP/TASS queries are stored in E_0.  Before routing them
            # together with carried state, express them in the current ego
            # frame.  ``cumulative`` is T(E_t -> E_0).
            new_points_current = self.ego_warp.t0_to_next(
                new_points, cumulative)
            state_feat = torch.cat([state_feat, new_feat], dim=1)
            state_points = torch.cat([state_points, new_points_current], dim=1)
            state_semantics = torch.cat([state_semantics, new_semantics], dim=1)
            current_metadata = (role_metadata_at(interval)
                                if self.training else None)
            # Only carried Queries belong to the current state ``t``.  A
            # stamp ``t+1`` RAP Query enters with actor ID -1 and receives its
            # first association only after the transition has produced
            # P_(t+1), using future metadata.  Existing carried IDs remain
            # immutable across prediction drift.
            if state_point_actor_id is None:
                association = self._initialize_actor_association(
                    decode_points(
                        state_points[:, :num_carried], self.pc_range),
                    current_metadata)
                state_point_actor_id = association['point_actor_id']
            if num_new:
                unassociated_new = torch.full(
                    (batch_size, num_new, self.num_refines), -1,
                    device=state_points.device, dtype=torch.long)
                state_point_actor_id = torch.cat([
                    state_point_actor_id, unassociated_new], dim=1)
            current_actor_valid = self._refresh_actor_association(
                state_point_actor_id, current_metadata)
            current_query_actor_id = self._query_actor_ids(
                state_point_actor_id)
            input_point_actor_id = state_point_actor_id.clone()
            source = state_points.new_zeros(batch_size, num_carried + num_new, 1)
            source[:, :num_carried] = 1
            activation = torch.cat([
                activation,
                activation.new_full((batch_size, num_new), interval + 1)
            ], dim=1)
            conditioned_feat = state_feat + self.source_embedding(
                source.squeeze(-1).long()) + self.activation_embedding(activation)
            role_prior = None
            role_prior_valid = None
            if state_role is not None:
                role_prior = torch.cat([
                    state_role,
                    state_points.new_zeros(batch_size, num_new,
                                           self.num_refines, 1)
                ], dim=1)
                role_prior_valid = torch.cat([
                    torch.ones_like(state_role, dtype=torch.bool),
                    torch.zeros(batch_size, num_new, self.num_refines, 1,
                                device=state_points.device, dtype=torch.bool)
                ], dim=1)
            teacher_role = None
            teacher_valid = None
            role_cache = None
            if self.training:
                # Routing is conditioned on the current state E_t.  Match
                # carried points against actors at t (not t+1); applying the
                # next ego transform here would supervise the wrong frame.
                metadata = current_metadata
                current_semantics = kwargs.get('voxel_semantics_current')
                if interval > 0:
                    temporal = kwargs.get('temporal_semantics')
                    if isinstance(temporal, dict):
                        current_semantics = temporal.get(
                            interval, temporal.get(str(interval)))
                    elif temporal is not None and len(temporal) >= interval:
                        current_semantics = temporal[interval - 1]
                    if isinstance(current_semantics, dict):
                        current_semantics = current_semantics.get(
                            'voxel_semantics')
                if current_semantics is not None and hasattr(
                        self.pts_bbox_head, 'build_role_match_cache'):
                    role_cache = self.pts_bbox_head.build_role_match_cache(
                        state_points, current_semantics,
                        role_metadata=metadata)
                    role_cache = self._apply_actor_association_to_role_cache(
                        role_cache, state_point_actor_id, metadata)
                    role_target = role_cache['role_target'].unsqueeze(-1)
                    role_target_valid = role_cache['role_valid'].unsqueeze(-1)
                elif metadata is not None:
                    role_target, role_target_valid = self._role_targets_from_metadata(
                        decode_points(state_points, self.pc_range), metadata)
                    role_cache = dict(role_target=role_target.squeeze(-1),
                                      role_valid=role_target_valid.squeeze(-1))
                    role_cache = self._apply_actor_association_to_role_cache(
                        role_cache, state_point_actor_id, metadata)
                    role_target = role_cache['role_target'].unsqueeze(-1)
                    role_target_valid = role_cache['role_valid'].unsqueeze(-1)
                role_cache = self._mask_new_query_role_cache(
                    role_cache, num_carried)
                if role_cache is not None:
                    role_target = role_cache['role_target'].unsqueeze(-1)
                    role_target_valid = role_cache['role_valid'].unsqueeze(-1)
                if role_cache is not None and self.prescf_role_teacher_forcing > 0:
                    teacher_role = role_target
                    teacher_valid = role_target_valid.clone()
                    # New Queries are invalid in the current cache and hence
                    # never receive GT_t teacher routing.  Their next role is
                    # supervised after P_(t+1) is available.
            role = self.role_router(
                conditioned_feat, state_points, state_semantics, source,
                role_prior=role_prior, role_prior_valid=role_prior_valid,
                teacher_role=teacher_role, teacher_valid=teacher_valid,
                teacher_forcing_ratio=(self.prescf_role_teacher_forcing
                                       if self.training else 0.0))
            pred_query_role = role['pred_query_role']
            route_query_role = role['route_query_role']

            # Planning and ego motion read the dynamic/static Query views
            # independently.  The same fused context drives the sole pose
            # head; planning displacement is derived from its cumulative
            # transform below in the VAD LiDAR coordinate convention.
            ego_point = state_points.new_full((batch_size, 1, 3), 0.5)
            dynamic_query_feat = route_query_role * conditioned_feat
            static_query_feat = (1.0 - route_query_role) * conditioned_feat
            ego_dynamic, _ = self.ego_cross_attn_dynamic(
                ego_point, ego_feat, state_points, dynamic_query_feat)
            ego_static, _ = self.ego_cross_attn_static(
                ego_point, ego_feat, state_points, static_query_feat)
            ego_context = self.ego_fusion_norm(
                ego_feat + self.ego_dynamic_proj(ego_dynamic - ego_feat) +
                self.ego_static_proj(ego_static - ego_feat))
            pose_raw_full = self.prescf_ego_pose_head(ego_context)
            pose_raw = pose_raw_full.squeeze(1)
            yaw = F.normalize(pose_raw[..., 2:4], dim=-1, eps=1e-6)
            pose = torch.cat([pose_raw[..., :2], yaw], dim=-1)
            predicted_next_to_current = self.ego_warp.pose_to_matrix(pose)
            # The pose head owns the only translation prediction.  Convert
            # its recursive ego transform to the VAD planning convention:
            # adjacent LiDAR-origin displacement in fixed E0-LiDAR axes.
            predicted_cumulative = self.ego_warp.compose(
                predicted_cumulative, predicted_next_to_current)
            predicted_lidar_to_t0 = self.ego_warp.ego_cumulative_to_lidar(
                predicted_cumulative, lidar_to_ego)
            lidar_position = predicted_lidar_to_t0[..., :3, 3]
            pred_traj = (lidar_position - previous_lidar_position)[
                ..., :2].unsqueeze(1)
            pred_trajs.append(pred_traj)
            previous_lidar_position = lidar_position
            next_to_current = predicted_next_to_current
            gt_pose = (adjacent_targets[interval]
                       if self.training and interval < len(adjacent_targets)
                       else None)
            teacher_forcing_mask = torch.zeros(
                batch_size, device=state_points.device, dtype=torch.bool)
            if self.training and gt_pose is not None and \
                    self.prescf_ego_teacher_forcing > 0:
                teacher_forcing_mask = (
                    torch.rand(batch_size, device=state_points.device) <
                    self.prescf_ego_teacher_forcing)
                next_to_current = torch.where(
                    teacher_forcing_mask[:, None, None], gt_pose,
                    predicted_next_to_current)
            cumulative = self.ego_warp.compose(cumulative, next_to_current)

            evolved = self.dual_evolution(
                conditioned_feat, state_points[:, :num_carried],
                new_points, ego_context, route_query_role,
                next_to_current, cumulative,
                self.ego_warp)
            # Execute every PreSCF block in every stage so DDP observes a
            # complete parameter graph.  Stage 1/2 use an identity blend for
            # the not-yet-enabled block instead of bypassing it entirely;
            # this gives those parameters a well-defined zero gradient and
            # avoids ``find_unused_parameters=True`` being required merely
            # because of the curriculum.
            interaction = self.dual_interaction(
                conditioned_feat, route_query_role,
                evolved['points_metric'])
            interaction_gate = interaction_stage_gate
            interaction_dynamic = dynamic_query_feat + interaction_gate * (
                interaction['dynamic_feat'] - dynamic_query_feat)
            interaction_static = static_query_feat + interaction_gate * (
                interaction['static_feat'] - static_query_feat)
            joint_raw = self.joint_refine(
                conditioned_feat, interaction_dynamic, interaction_static,
                evolved['points_metric'])
            joint_gate = joint_stage_gate
            joint = dict(
                query_feat=conditioned_feat + joint_gate * (
                    joint_raw['query_feat'] - conditioned_feat),
                point_correction=joint_gate * joint_raw['point_correction'],
                role_correction=joint_gate * joint_raw['role_correction'])
            correction_min_gate = float(self.dsqe_cfg.get(
                'joint_correction_min_gate', 0.1))
            correction_gate = (correction_min_gate +
                               (1.0 - correction_min_gate) *
                               route_query_role).unsqueeze(2)
            final_metric = (evolved['points_metric'] +
                            correction_gate * joint['point_correction'])
            correction_scale = float(self.dsqe_cfg.get(
                'role_correction_scale', 0.25))
            next_role = (
                role['route_role'] + correction_scale *
                joint['role_correction'].tanh()).clamp(
                    self.role_router.eps, 1.0 - self.role_router.eps)
            next_role_logits = torch.logit(
                next_role, eps=self.role_router.eps)
            next_query_role = (
                role['pool_weights'] * next_role).sum(dim=2)
            final_points = encode_points(final_metric, self.pc_range)
            semantics = self.joint_refine.predict_semantics(
                joint['query_feat'], final_metric)

            if self.training:
                forecast_masks.append(self._prescf_foreground_mask(
                    final_points, interval, img_metas, kwargs))
            forecast_points.append(final_points)
            forecast_semantics.append(semantics)
            match_cache = None
            future_metadata = (role_metadata_at(interval + 1)
                               if self.training else None)
            temporal_semantics = kwargs.get('temporal_semantics')
            if self.training and temporal_semantics is not None and \
                    hasattr(self.pts_bbox_head, 'build_future_match_cache'):
                if isinstance(temporal_semantics, dict):
                    future = temporal_semantics.get(
                        interval + 1, temporal_semantics.get(str(interval + 1)))
                else:
                    future = temporal_semantics[interval]
                if isinstance(future, dict):
                    future = future.get('voxel_semantics')
                if future is not None:
                    match_cache = self.pts_bbox_head.build_future_match_cache(
                        final_points, future,
                        role_metadata=future_metadata)
            recovery = self._recover_actor_association(
                state_point_actor_id, final_metric, future_metadata,
                max_distance=self.prescf_actor_recovery_max_distance)
            state_point_actor_id = recovery['point_actor_id']
            if match_cache is not None:
                match_cache = self._apply_actor_association_to_role_cache(
                    match_cache, state_point_actor_id, future_metadata)
            match_cache_list.append(match_cache)
            future_actor_valid = recovery['actor_valid']
            future_query_actor_id = recovery['query_actor_id']
            role_loss = final_metric.new_zeros(())
            dynamic_loss = final_metric.new_zeros(())
            static_loss = final_metric.new_zeros(())
            motion_valid = source.bool()
            if previous_motion is not None and num_carried:
                overlap = min(previous_motion.shape[1],
                              evolved['query_motion'].shape[1])
                valid_overlap = (motion_valid[:, :overlap] &
                                 previous_motion_valid[:, :overlap])
                # Consecutive motion vectors live in consecutive future ego
                # frames.  Rotate the previous prediction into E_(t+1)
                # before imposing temporal continuity.
                current_from_previous = self.ego_warp.inverse(
                    next_to_current)[..., :3, :3]
                previous_aligned = torch.einsum(
                    'bnj,bij->bni', previous_motion[:, :overlap],
                    current_from_previous)
                delta_motion = (evolved['query_motion'][:, :overlap] -
                                previous_aligned).abs().mean(-1)
                smooth_loss = (delta_motion * valid_overlap.squeeze(-1).to(
                    delta_motion.dtype)).sum() / valid_overlap.sum().clamp_min(1)
            else:
                smooth_loss = final_metric.new_zeros(())
            previous_motion = evolved['query_motion']
            previous_motion_valid = motion_valid
            static_temporal_target = None
            if gt_pose is not None and num_carried:
                # Static carried points should follow the GT ego warp only;
                # new Queries have no historical state and are deliberately
                # excluded from this temporal consistency target.
                input_carried_metric = decode_points(
                    state_points[:, :num_carried], self.pc_range)
                static_temporal_target = self.ego_warp.transform_metric(
                    input_carried_metric,
                    self.ego_warp.inverse(gt_pose))
            prescf_outputs.append(dict(
                role=role,
                role_logits=role['role_logits'], role_pred=role['role_pred'],
                pred_query_role=pred_query_role,
                query_role=pred_query_role,
                route_role=role['route_role'],
                route_query_role=route_query_role,
                next_role_logits=next_role_logits,
                next_role_pred=next_role,
                next_query_role=next_query_role,
                role_prior=role_prior,
                role_prior_valid=role_prior_valid,
                query_motion=evolved['query_motion'],
                motion_valid=motion_valid,
                predicted_pose=pose,
                predicted_relative_matrix=predicted_next_to_current,
                routed_relative_matrix=next_to_current,
                ego_teacher_forcing_mask=teacher_forcing_mask,
                gt_pose_target=(gt_pose.detach() if gt_pose is not None else None),
                input_feat=state_feat, input_points=state_points,
                input_semantics=state_semantics,
                input_point_actor_id=input_point_actor_id,
                input_query_actor_id=current_query_actor_id,
                current_actor_valid=current_actor_valid,
                points_metric=final_metric, evolved_points_metric=evolved['points_metric'],
                carried_prior_metric=evolved['carried_prior_metric'],
                carried_evolved_metric=final_metric[:, :num_carried],
                static_temporal_target_metric=static_temporal_target,
                semantics=semantics, joint_feat=joint['query_feat'],
                loss_role=role_loss, loss_dynamic=dynamic_loss,
                loss_static=static_loss, loss_smooth=smooth_loss,
                num_carried=num_carried,
                current_role_cache=role_cache,
                future_match_cache=match_cache,
                point_actor_id=state_point_actor_id,
                query_actor_id=future_query_actor_id,
                actor_valid=future_actor_valid,
                actor_recovered_mask=recovery['recovered_mask'],
                pool_weights=role['pool_weights']))
            state_feat, state_points, state_semantics = (
                joint['query_feat'], final_points, semantics)
            state_role = next_role
        # PreSCF deliberately exposes only the active curriculum horizon; do
        # not append a frozen BaseLine trajectory head to fill missing steps.
        if self.dsqe_mode != 'prescf' and not self.pretrain and \
                len(pred_trajs) < self.num_fu_frames:
            pred_trajs.append(self.traj_head(ego_feat))
        return dict(cls_score=all_semantics[:, current_mask],
                    refine_pts=all_points[:, current_mask], outs=outs,
                    forecast_semantics_list=forecast_semantics,
                    forecast_points_list=forecast_points,
                    pred_trajs_list=pred_trajs,
                    forecast_points_mask_list=forecast_masks,
                    prescf_outputs=prescf_outputs,
                    match_cache_list=match_cache_list)

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        img = [img] if img is None else img

        result = self.simple_test(img_metas[0], img[0], **kwargs)



        return result


    def forward_backbone(self,img,img_metas,**kwargs):
        B = img.shape[0]
        if self.dsqe_mode == 'prescf':
            self._ensure_tass_state()
        ego_states_all = kwargs['temporal_ego_states']
        if isinstance(ego_states_all, dict):
            ego_states = ego_states_all.get(0, next(iter(ego_states_all.values())))
        else:
            ego_states = ego_states_all[0]
        if isinstance(ego_states, dict):
            ego_states = ego_states.get(0, next(iter(ego_states.values())))
        ego_states = ego_states.to(device=img.device, dtype=torch.float32)
        ego_feat = self.plan_head(ego_states.reshape(B, 1, -1))
        points_scale = self.points_scale_branch(ego_feat).tanh()
        self.pts_bbox_head.points_scale = (points_scale + 1) / 2 * (1.5 - 0.8) + 0.8
        if self.training:
            outs = self.pts_bbox_head(self.extract_feat(img, img_metas), img_metas)
        else:
            outs = self.simple_test_online(img_metas, img)
        if self.dsqe_mode == 'prescf':
            return self._forward_prescf(outs, ego_feat, img_metas, kwargs)
        return self._forward_baseline_scf(outs, ego_feat, img_metas, kwargs)

    def simple_test(self,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""

        for key in kwargs.keys():
            kwargs[key] = kwargs[key][0]

        outputs = self.forward_backbone(img, img_metas, **kwargs)
        cls_score, curr_query_pos, outs = outputs['cls_score'],outputs['refine_pts'],outputs['outs']

        pred_dict = dict(cls_scores=outs['all_cls_scores'][-1][:,self.pts_bbox_head.ind_stamps_all==0], refine_pts=outs['all_refine_pts'][-1][:,self.pts_bbox_head.ind_stamps_all==0])
        occ_pred = self.pts_bbox_head.get_occ(pred_dict)[0]
        # self.pred_num += torch.bincount(occ_pred.flatten())
        geo_pred = torch.ones_like(occ_pred) * 17
        geo_pred[occ_pred != 17] = 0
        res_dict = {f'semantic_occ_0s': [occ_pred.cpu().numpy()],
                    f'geo_occ_0s': [geo_pred.cpu().numpy()]}

        forecast_points_list, forecast_semantics_list, pred_trajs_list = \
            outputs['forecast_points_list'],outputs['forecast_semantics_list'],outputs['pred_trajs_list']

        for interval in range(self.num_fu_frames):
            input_dict = dict(cls_scores=forecast_semantics_list[interval],
                              refine_pts=forecast_points_list[interval])
            occ_forecast = self.pts_bbox_head.get_occ(input_dict)[0]  # eval for single batch
            geo_forecast = torch.ones_like(occ_forecast) * 17
            geo_forecast[occ_forecast != 17] = 0
            # pred_traj_list.append(pred_traj)
            res_dict.update({
                f'semantic_occ_{int(interval + 1)}s': [occ_forecast.cpu().numpy()],
                f'geo_occ_{int(interval + 1)}s': [geo_forecast.cpu().numpy()],
            })

        res_dict['pred_traj'] = torch.cat(pred_trajs_list, 1)
        return res_dict

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      **kwargs):

        temporal_semantics = kwargs['temporal_semantics']
        B = img.shape[0]
        temporal2ego = kwargs['temporal2ego']
        kwargs['voxel_semantics_current'] = voxel_semantics
        if kwargs.get('temporal_agent_lidar2ego') is None and img_metas:
            # Backward-compatible fallback for private pipelines that did not
            # add the explicit calibration key to Collect4D.
            matrices = []
            for meta in img_metas:
                ego2lidar = meta.get('ego2lidar') if isinstance(meta, dict) else None
                if ego2lidar is None:
                    matrices = []
                    break
                matrices.append(torch.linalg.inv(torch.as_tensor(
                    ego2lidar, device=img.device, dtype=torch.float32)))
            if matrices:
                kwargs['temporal_agent_lidar2ego'] = torch.stack(matrices)
        outputs = self.forward_backbone(img,img_metas,**kwargs)
        cls_score,refine_pts,outs = outputs['cls_score'],outputs['refine_pts'],outputs['outs']

        losses = dict()
        ind_stamps_all = self.pts_bbox_head.ind_stamps_all
        if self.pretrain:
            loss_inputs = [voxel_semantics, temporal_semantics, temporal2ego, outs]
            losses.update(self.pts_bbox_head.loss_pretrain(*loss_inputs))
        else:
            # outs_inits = dict(init_points = outs['init_points'],all_cls_scores = [], all_refine_pts = [])
            loss_inputs = [voxel_semantics, temporal_semantics, temporal2ego, outs]
            losses.update(self.pts_bbox_head.loss_pretrain(*loss_inputs))
            outs['init_points'] = None
            for i in range(len(outs['all_cls_scores'])):
                outs['all_cls_scores'][i] = outs['all_cls_scores'][i][:,ind_stamps_all==0]
                outs['all_refine_pts'][i] = outs['all_refine_pts'][i][:,ind_stamps_all==0]
            loss_inputs = [voxel_semantics,outs,]
            losses.update(self.pts_bbox_head.loss(*loss_inputs))

        forecast_points_list = outputs['forecast_points_list']
        forecast_semantics_list = outputs['forecast_semantics_list']
        pred_trajs_list = outputs['pred_trajs_list']
        forecast_points_mask_list = outputs['forecast_points_mask_list']

        num_fu_frames = len(forecast_semantics_list)
        temporal_semantics = kwargs['temporal_semantics']
        if isinstance(temporal_semantics, dict):
            voxel_semantics_temporal = []
            for interval in range(1, num_fu_frames + 1):
                item = temporal_semantics.get(
                    interval, temporal_semantics.get(str(interval)))
                if item is None:
                    raise KeyError(
                        'temporal_semantics is missing future step {}'.format(
                            interval))
                voxel_semantics_temporal.append(
                    item['voxel_semantics'] if isinstance(item, dict) else item)
        else:
            voxel_semantics_temporal = [
                item['voxel_semantics'] if isinstance(item, dict) else item
                for item in temporal_semantics[:num_fu_frames]]
        losses.update(
            self.pts_bbox_head.loss_future(voxel_semantics_temporal,
                                           forecast_points_list,forecast_semantics_list,
                                           forecast_points_mask_list,
                                           dsqe_outputs=(outputs.get('prescf_outputs')
                                                         if self.dsqe_mode == 'prescf' else None),
                                           match_cache_list=outputs.get(
                                               'match_cache_list')))
        if self.dsqe_mode == 'prescf' and outputs.get('prescf_outputs'):
            for interval, state in enumerate(outputs['prescf_outputs'], 1):
                current_role_cache, future_match_cache = \
                    self._prescf_supervision_caches(state)
                pred_role_output, next_role_output = \
                    self._prescf_role_supervision_outputs(state)
                # rho_t describes E_t/P_t/Z_t and is therefore supervised
                # only by the current-state cache.  The cache generated from
                # P_(t+1) supervises next_role and future geometry/semantics,
                # but can never feed back into the target for route rho_t.
                pred_role_loss = (self.pts_bbox_head._loss_role(
                                 pred_role_output, current_role_cache)
                             if current_role_cache is not None and hasattr(
                                 self.pts_bbox_head, '_loss_role') else
                             state['loss_role'])
                next_role_loss = (self.pts_bbox_head._loss_role(
                                      next_role_output, future_match_cache)
                                  if future_match_cache is not None and hasattr(
                                      self.pts_bbox_head, '_loss_role') else
                                  state['loss_role'] * 0)
                role_loss = (pred_role_loss +
                             self.prescf_next_role_loss_weight *
                             next_role_loss)
                losses['fu{}.loss_role'.format(interval)] = (
                    self.prescf_loss_weights['role'] * role_loss)
                losses['fu{}.role_pred_term'.format(interval)] = \
                    pred_role_loss.detach()
                losses['fu{}.role_next_term'.format(interval)] = \
                    next_role_loss.detach()
                if current_role_cache is not None and hasattr(
                        self.pts_bbox_head, '_role_metrics'):
                    for metric_name, metric_value in self.pts_bbox_head._role_metrics(
                            pred_role_output, current_role_cache).items():
                        losses['fu{}.{}'.format(interval, metric_name)] = metric_value
                if future_match_cache is not None and hasattr(
                        self.pts_bbox_head, '_role_metrics'):
                    for metric_name, metric_value in self.pts_bbox_head._role_metrics(
                            next_role_output, future_match_cache).items():
                        losses['fu{}.next_{}'.format(
                            interval, metric_name)] = metric_value
                if state.get('gt_pose_target') is not None:
                    target_pose = self.ego_warp.matrix_to_pose(
                        state['gt_pose_target'])
                    losses['fu{}.loss_ego'.format(interval)] = (
                        self.prescf_loss_weights['ego'] * F.smooth_l1_loss(
                            state['predicted_pose'],
                            target_pose.to(
                                state['predicted_pose'].dtype)))
                else:
                    losses['fu{}.loss_ego'.format(interval)] = (
                        state['predicted_pose'].sum() * 0)
                refine_metric = decode_points(
                    forecast_points_list[interval - 1].reshape(
                        forecast_points_list[interval - 1].shape[0], -1, 3),
                    self.pc_range)
                static_loss = (self.pts_bbox_head._loss_static(
                    state, current_role_cache, refine_metric)
                               if current_role_cache is not None and hasattr(
                                   self.pts_bbox_head, '_loss_static') else
                               state['loss_static'])
                losses['fu{}.loss_static'.format(interval)] = (
                    self.prescf_loss_weights['static'] * static_loss)
                if future_match_cache is not None and hasattr(
                        self.pts_bbox_head, '_loss_dynamic'):
                    dynamic = self.pts_bbox_head._loss_dynamic(
                        forecast_semantics_list[interval - 1], state,
                        future_match_cache, refine_metric,
                        return_diagnostics=True)
                    dynamic_loss = dynamic['loss']
                    # Diagnostic keys intentionally omit the ``loss`` token:
                    # MMDetection sums every dict entry containing "loss".
                    # Adding detached loss-named values would silently change
                    # the optimization objective through double counting.
                    for name in (
                            'dynamic_pred_to_gt', 'dynamic_gt_to_pred',
                            'dynamic_semantic'):
                        value = dynamic['loss_' + name]
                        losses['fu{}.{}'.format(interval, name)] = value.detach()
                    for name in (
                            'dynamic_pred_valid_ratio',
                            'dynamic_gt_covered_ratio'):
                        losses['fu{}.{}'.format(interval, name)] = \
                            dynamic[name].detach()
                else:
                    dynamic_loss = state['loss_dynamic']
                losses['fu{}.loss_dynamic'.format(interval)] = (
                    self.prescf_loss_weights['dynamic'] * dynamic_loss)
                losses['fu{}.loss_smooth'.format(interval)] = (
                    self.prescf_loss_weights['smooth'] * state['loss_smooth'])
                leak_loss = (self.pts_bbox_head._loss_leak(
                    state, current_role_cache)
                    if current_role_cache is not None and hasattr(
                        self.pts_bbox_head, '_loss_leak') else
                    state['loss_dynamic'] * 0)
                losses['fu{}.loss_leak'.format(interval)] = (
                    self.prescf_loss_weights['leak'] * leak_loss)
        for interval,pred_traj in enumerate(pred_trajs_list):
            if interval >= kwargs['temporal_trajs'].shape[1]:
                break
            loss_traj = self.loss_traj(pred_traj.squeeze(1), kwargs['temporal_trajs'][:, interval, :], interval + 1)
            losses.update(loss_traj)

        return losses
