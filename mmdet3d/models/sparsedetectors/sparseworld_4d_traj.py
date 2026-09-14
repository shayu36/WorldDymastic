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
            new_residual_scale=cfg.get('new_residual_scale', 1.0))
        self.dual_interaction = DSQEDualInteraction(
            self.out_dim, num_heads=cfg.get('num_heads', 8),
            local_k=cfg.get('local_k', 16),
            dropout=cfg.get('dropout', drop_out),
            dynamic_from_static_init=cfg.get('lambda_DS', 1.0),
            static_from_dynamic_init=cfg.get('lambda_SD', 0.25))
        self.joint_refine = DSQEJointRefine(
            self.out_dim, self.num_refines, num_classes=17)
        self.source_embedding = nn.Embedding(2, self.out_dim)
        self.activation_embedding = nn.Embedding(
            self.num_fu_frames + 1, self.out_dim)
        self.prescf_ego_pose_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim), nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, 4))
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
            smooth=float(cfg.get('lambda_smooth', 0.01)))
        nn.init.zeros_(self.prescf_ego_pose_head[-1].weight)
        with torch.no_grad():
            self.prescf_ego_pose_head[-1].bias.copy_(
                torch.tensor([0.0, 0.0, 0.0, 1.0]))
        # Snapshot for reducing only rank-local TASS count increments.  It is
        # intentionally non-persistent and initialized after checkpoint load
        # on the first distributed forward.
        self._prescf_synced_tass_counts = None
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

    def init_weights(self):
        self.pts_bbox_head.init_weights()
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)
        # Warm-start the absolute PreSCF semantic head from the trained
        # BaseLine classifier.  The BaseLine classifier emits one 17-way
        # vector per refine point; averaging those point-specific projections
        # gives the new point-conditioned head a sensible initial prior.
        if self.dsqe_mode == 'prescf' and hasattr(self, 'joint_refine'):
            source = self.cls_branch[-1]
            target = self.joint_refine.semantic_head
            with torch.no_grad():
                if source.weight.shape[0] % target.out_features == 0 and \
                        source.weight.shape[1] == target.in_features:
                    weights = source.weight.reshape(
                        -1, target.out_features, target.in_features).mean(0)
                    bias = source.bias.reshape(
                        -1, target.out_features).mean(0)
                    target.weight.copy_(weights)
                    target.bias.copy_(bias)

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
            target_weight = prefix + 'joint_refine.semantic_head.weight'
            target_bias = prefix + 'joint_refine.semantic_head.bias'
            source_index = len(self.cls_branch) - 1
            source_weight = prefix + 'cls_branch.{}.weight'.format(source_index)
            source_bias = prefix + 'cls_branch.{}.bias'.format(source_index)
            if target_weight not in state_dict and source_weight in state_dict:
                weight = state_dict[source_weight]
                classes = self.joint_refine.semantic_head.out_features
                if weight.shape[0] % classes == 0:
                    state_dict[target_weight] = weight.reshape(
                        -1, classes, weight.shape[-1]).mean(0)
            if target_bias not in state_dict and source_bias in state_dict:
                bias = state_dict[source_bias]
                classes = self.joint_refine.semantic_head.out_features
                if bias.shape[0] % classes == 0:
                    state_dict[target_bias] = bias.reshape(-1, classes).mean(0)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def set_epoch(self, epoch):
        self.curr_epoch = epoch
        if self.dsqe_mode == 'prescf':
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
                layers = getattr(getattr(self.pts_bbox_head, 'transformer', None),
                                 'decoder', None)
                layers = getattr(layers, 'decoder_layers', None)
                if layers is not None:
                    for layer in list(layers)[-self._prescf_unfreeze_tass_layers:]:
                        for parameter in layer.parameters():
                            parameter.requires_grad = True
        if self.dsqe_mode == 'prescf':
            # PreSCF is initialized from the finished epoch-56 BaseLine.  Its
            # epoch 0 is already a future-state fine-tuning stage; inheriting
            # the historical BaseLine ``pretrain`` flag would multiply every
            # future occupancy loss by zero and leave the absolute semantic
            # head without gradients.
            self.pretrain = False
            self.pts_bbox_head.pretrain = False
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
        if hasattr(value, 'data') and not torch.is_tensor(value):
            value = value.data
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, (list, tuple)):
            return torch.as_tensor(value, device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _to_batch_sequence(value, device, dtype):
        if hasattr(value, 'data') and not torch.is_tensor(value):
            value = value.data
        if isinstance(value, (list, tuple)):
            return [SparseWorld4DTraj._to_batch_tensor(item, device, dtype)
                    for item in value]
        tensor = SparseWorld4DTraj._to_batch_tensor(value, device, dtype)
        if tensor.ndim >= 3:
            return [tensor[index] for index in range(tensor.shape[0])]
        return [tensor]

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
        if getattr(head, 'ind_stamps_all', None) is None:
            stamps = head.num_stamps_all.float()
            stamps = stamps / stamps.sum(-1, keepdim=True).clamp_min(1e-6)
            head.ind_stamps_all = get_matched_inds(
                stamps, [self.num_query] + self.num_fu_query)
            if hasattr(head, 'reset_mask'):
                head.reset_mask()
        if getattr(self, 'dsqe_enabled', False):
            self._synchronize_tass_state(self)

    def _build_role_metadata(self, kwargs, interval, gt_relative_matrices=None):
        """Build actor role targets in the *future ego frame*.

        The temporal nuScenes cache stores ``gt_agent_fut_trajs`` as the
        displacement at each horizon relative to the current ego frame
        (``E_0``), rather than as increments between consecutive horizons.
        We therefore select the displacement at ``interval`` directly and
        transform the resulting actor centre with ``T(E_0 -> E_interval)``.
        Older/private caches that contain adjacent increments can opt into the
        legacy accumulation behaviour with
        ``dsqe_cfg.role_trajectory_mode='increment'``.
        """
        boxes = kwargs.get('temporal_agent_boxes')
        feats = kwargs.get('temporal_agent_feats')
        labels = kwargs.get('temporal_agent_labels')
        if boxes is None:
            return None
        if hasattr(boxes, 'data') and not torch.is_tensor(boxes):
            boxes = boxes.data
        if hasattr(feats, 'data') and not torch.is_tensor(feats):
            feats = feats.data
        if hasattr(labels, 'data') and not torch.is_tensor(labels):
            labels = labels.data
        device = (gt_relative_matrices.device if torch.is_tensor(
            gt_relative_matrices) else (boxes.device if torch.is_tensor(boxes)
                                         else torch.device('cpu')))
        box_list = self._to_batch_sequence(boxes, device, torch.float32)
        feat_list = None if feats is None else self._to_batch_sequence(
            feats, device, torch.float32)
        label_list = None if labels is None else self._to_batch_sequence(
            labels, device, torch.long)
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
            'role_trajectory_mode', 'absolute')).lower()
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
            # The stored actor future trajectory is indexed by horizon and is
            # expressed in E_0.  ``increment`` remains available for caches
            # generated by older private preprocessing scripts.
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
                        future_valid[:n] = mask_values[:, step] > 0.5
                    yaw_start = agent_feat.shape[-1] - self.num_fu_frames
                    if interval > 0 and yaw_start >= mask_end and n:
                        yaw_values = agent_feat[
                            :n, yaw_start:yaw_start + self.num_fu_frames]
                        yaw_values = torch.atan2(yaw_values.sin(),
                                                yaw_values.cos())
                        if trajectory_mode == 'absolute':
                            yaw_delta[:n] = yaw_values[:, step]
                        else:
                            yaw_delta[:n] = yaw_values[:, :step + 1].sum(1)
            future_centers_t0 = centers.clone()
            if interval > 0 and displacement.shape[0] == num_agents:
                future_centers_t0[:, :2] += displacement
            future_centers = self.ego_warp.transform_metric(
                future_centers_t0.unsqueeze(0).unsqueeze(2),
                to_future[b:b + 1])[0, :, 0]
            transform_yaw = torch.atan2(to_future[b, 1, 0],
                                        to_future[b, 0, 0])
            future_yaw = (current[:, 6] if current.shape[-1] >= 7 else
                          centers.new_zeros(num_agents)) + yaw_delta + transform_yaw
            if current.shape[-1] >= 6:
                # ``traj_info`` is produced from the native nuScenes boxes,
                # whose size tuple is ``(w, l, h)``.  Role matching operates
                # in an ego-frame box convention with the x axis along the
                # box length, so make the ordering explicit instead of
                # silently swapping an actor's footprint axes.  Projects
                # that already converted their cache can opt into ``lhw``.
                raw_dims = current[:, 3:6].abs()
                dims_order = str(self.dsqe_cfg.get(
                    'role_box_dims_order', 'wlh')).lower()
                if dims_order == 'wlh':
                    dims = raw_dims[:, [1, 0, 2]]
                elif dims_order in ('lhw', 'lwh'):
                    dims = raw_dims
                else:
                    raise ValueError(
                        'role_box_dims_order must be wlh or lhw/lwh, got {}'.format(
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
                actor_role = torch.zeros(num_agents, device=device,
                                         dtype=torch.float32)
                actor_role[torch.tensor(
                    [int(x) in dynamic_ids for x in actor_labels.tolist()],
                    device=device, dtype=torch.bool)] = 1.0
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
                dims=dims, yaw=future_yaw)
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
                matched &= (local_x.abs() <= dims[index, 0] * 0.5 + 0.5)
                matched &= (local_y.abs() <= dims[index, 1] * 0.5 + 0.5)
            target[b].reshape(-1, 1)[matched] = role[index[matched], None]
            valid[b].reshape(-1, 1)[matched] = True
        return target, valid

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
        the old residual-path convention.  The encoded point range test is
        the same fallback used by BaseLine when no trajectory metadata is
        available, while remaining frame-consistent for every future step.
        """
        return points[..., 0] >= 0

    @staticmethod
    def _matrix_value(value, device, dtype, batch_size):
        """Normalize one collated transform value to ``[B, 4, 4]``."""
        if hasattr(value, 'data') and not torch.is_tensor(value):
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
        if hasattr(sequence, 'data') and not torch.is_tensor(sequence):
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
        activation = state_points.new_zeros(
            batch_size, state_points.shape[1], dtype=torch.long)
        cumulative = self.ego_warp.identity(
            batch_size, state_points.device, state_points.dtype)
        forecast_points, forecast_semantics = [], []
        pred_trajs, forecast_masks, prescf_outputs, match_cache_list = [], [], [], []
        num_forecast = (self._num_forecast_frames()
                        if hasattr(self, '_num_forecast_frames') else
                        self.num_fu_frames)
        adjacent_targets = (self._build_adjacent_ego_targets(
            kwargs, batch_size, num_forecast, state_points.device,
            state_points.dtype) if self.training else [])
        previous_motion = None
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
                metadata = self._build_role_metadata(kwargs, interval)
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
                    role_target = role_cache['role_target'].unsqueeze(-1)
                    role_target_valid = role_cache['role_valid'].unsqueeze(-1)
                elif metadata is not None:
                    role_target, role_target_valid = self._role_targets_from_metadata(
                        decode_points(state_points, self.pc_range), metadata)
                    role_cache = dict(role_target=role_target.squeeze(-1),
                                      role_valid=role_target_valid.squeeze(-1))
                if role_cache is not None and self.prescf_role_teacher_forcing > 0:
                    teacher_role = role_target
                    teacher_valid = role_target_valid.clone()
                    # New Queries have no reliable temporal identity.  They
                    # may be supervised by the future occupancy cache, but
                    # must not inherit a carried Query's GT role teacher.
                    if num_new and teacher_valid is not None:
                        teacher_valid[:, num_carried:] = False
            role = self.role_router(
                conditioned_feat, state_points, state_semantics, source,
                role_prior=role_prior, role_prior_valid=role_prior_valid,
                teacher_role=teacher_role, teacher_valid=teacher_valid,
                teacher_forcing_ratio=(self.prescf_role_teacher_forcing
                                       if self.training else 0.0))
            query_role = role['query_role']

            # Planning and ego motion use only the current predicted state.
            ego_point = state_points.new_full((batch_size, 1, 3), 0.5)
            ego_context, _ = self.ego_cross_attn(
                ego_point, ego_feat, state_points, conditioned_feat)
            pred_traj = self.traj_head(ego_context)
            pred_trajs.append(pred_traj)
            pose_raw = self.prescf_ego_pose_head(ego_context).squeeze(1)
            yaw = F.normalize(pose_raw[..., 2:4], dim=-1, eps=1e-6)
            pose = torch.cat([pose_raw[..., :2], yaw], dim=-1)
            predicted_next_to_current = self.ego_warp.pose_to_matrix(pose)
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
                new_points, ego_context, query_role,
                next_to_current, cumulative,
                self.ego_warp)
            interaction = self.dual_interaction(
                conditioned_feat, query_role, evolved['points_metric'])
            joint = self.joint_refine(
                conditioned_feat, interaction['dynamic_feat'],
                interaction['static_feat'], evolved['points_metric'])
            correction_gate = (0.25 + 0.75 * query_role).unsqueeze(2)
            final_metric = (evolved['points_metric'] +
                            correction_gate * joint['point_correction'])
            correction_scale = float(self.dsqe_cfg.get(
                'role_correction_scale', 0.25))
            corrected_role = (
                role['route_role'] + correction_scale *
                joint['role_correction'].tanh()).clamp(
                    self.role_router.eps, 1.0 - self.role_router.eps)
            corrected_query_role = (
                role['pool_weights'] * corrected_role).sum(dim=2)
            final_points = encode_points(final_metric, self.pc_range)
            semantics = self.joint_refine.predict_semantics(
                joint['query_feat'], final_metric)

            if self.training:
                forecast_masks.append(self._prescf_foreground_mask(
                    final_points, interval, img_metas, kwargs))
            forecast_points.append(final_points)
            forecast_semantics.append(semantics)
            match_cache = None
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
                    metadata = self._build_role_metadata(kwargs, interval + 1)
                    match_cache = self.pts_bbox_head.build_future_match_cache(
                        final_points, future, role_metadata=metadata)
            match_cache_list.append(match_cache)
            role_loss = final_metric.new_zeros(())
            dynamic_loss = final_metric.new_zeros(())
            static_loss = final_metric.new_zeros(())
            if previous_motion is not None and num_carried:
                overlap = min(previous_motion.shape[1],
                              evolved['query_motion'].shape[1])
                smooth_loss = (evolved['query_motion'][:, :overlap] -
                               previous_motion[:, :overlap]).abs().mean()
            else:
                smooth_loss = final_metric.new_zeros(())
            previous_motion = evolved['query_motion']
            prescf_outputs.append(dict(
                role=role, corrected_role=corrected_role,
                role_logits=role['role_logits'], role_pred=role['role_pred'],
                query_role=corrected_query_role, query_motion=evolved['query_motion'],
                predicted_pose=pose,
                predicted_relative_matrix=predicted_next_to_current,
                routed_relative_matrix=next_to_current,
                ego_teacher_forcing_mask=teacher_forcing_mask,
                gt_pose_target=(gt_pose.detach() if gt_pose is not None else None),
                input_feat=state_feat, input_points=state_points,
                input_semantics=state_semantics,
                points_metric=final_metric, evolved_points_metric=evolved['points_metric'],
                semantics=semantics, joint_feat=joint['query_feat'],
                loss_role=role_loss, loss_dynamic=dynamic_loss,
                loss_static=static_loss, loss_smooth=smooth_loss,
                num_carried=num_carried, match_cache=match_cache,
                role_cache=role_cache,
                pool_weights=role['pool_weights']))
            state_feat, state_points, state_semantics = (
                joint['query_feat'], final_points, semantics)
            state_role = corrected_role
        if not self.pretrain and len(pred_trajs) < self.num_fu_frames:
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

        voxel_semantics_temporal = [sem['voxel_semantics'] for sem in kwargs['temporal_semantics'].values()]

        num_fu_frames = len(forecast_semantics_list)
        losses.update(
            self.pts_bbox_head.loss_future(voxel_semantics_temporal[:num_fu_frames],
                                           forecast_points_list,forecast_semantics_list,
                                           forecast_points_mask_list,
                                           dsqe_outputs=(outputs.get('prescf_outputs')
                                                         if self.dsqe_mode == 'prescf' else None),
                                           match_cache_list=outputs.get(
                                               'match_cache_list')))
        if self.dsqe_mode == 'prescf' and outputs.get('prescf_outputs'):
            for interval, state in enumerate(outputs['prescf_outputs'], 1):
                cache = state.get('match_cache')
                # Role logits drive the transition E_t -> E_(t+1), so their
                # train target is the resulting future state.  This also
                # provides direct supervision for newly activated Queries;
                # current-state GT teacher forcing remains carried-only.
                role_cache = cache if cache is not None else state.get('role_cache')
                role_loss = (self.pts_bbox_head._loss_role(state, role_cache)
                             if role_cache is not None and hasattr(
                                 self.pts_bbox_head, '_loss_role') else
                             state['loss_role'])
                losses['fu{}.loss_role'.format(interval)] = (
                    self.prescf_loss_weights['role'] * role_loss)
                if role_cache is not None and hasattr(self.pts_bbox_head,
                                                 '_role_metrics'):
                    for metric_name, metric_value in self.pts_bbox_head._role_metrics(
                            state, role_cache).items():
                        losses['fu{}.{}'.format(interval, metric_name)] = metric_value
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
                    state, cache, refine_metric)
                               if cache is not None and hasattr(
                                   self.pts_bbox_head, '_loss_static') else
                               state['loss_static'])
                losses['fu{}.loss_static'.format(interval)] = (
                    self.prescf_loss_weights['static'] * static_loss)
                losses['fu{}.loss_dynamic'.format(interval)] = (
                    self.prescf_loss_weights['dynamic'] * (
                        self.pts_bbox_head._loss_dynamic(
                            forecast_semantics_list[interval - 1], state,
                            cache, refine_metric)
                        if cache is not None and hasattr(
                            self.pts_bbox_head, '_loss_dynamic') else
                        state['loss_dynamic']))
                losses['fu{}.loss_smooth'.format(interval)] = (
                    self.prescf_loss_weights['smooth'] * state['loss_smooth'])
        for interval,pred_traj in enumerate(pred_trajs_list):
            if interval >= kwargs['temporal_trajs'].shape[1]:
                break
            loss_traj = self.loss_traj(pred_traj.squeeze(1), kwargs['temporal_trajs'][:, interval, :], interval + 1)
            losses.update(loss_traj)

        return losses
