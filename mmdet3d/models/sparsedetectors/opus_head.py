import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops import knn, Voxelization
from mmdet.core import multi_apply
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmdet3d.models.sparsedetectors.bbox.utils import decode_points,dist_loss_weight,get_matched_inds
import torch

colors = torch.tensor([
    [1.00, 0.00, 0.00],   # 红 Red
    [1.00, 0.65, 0.00],   # 橙 Orange
    [1.00, 1.00, 0.00],   # 黄 Yellow
    [0.00, 0.50, 0.00],   # 绿 Green
    [0.00, 1.00, 1.00],   # 青 Cyan
    [0.00, 0.00, 1.00],   # 蓝 Blue
    [0.50, 0.00, 0.50],   # 紫 Purple
])


@HEADS.register_module()
class OPUSHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 num_fu_query,
                 num_fu_frames,
                 transformer=None,
                 pc_range=[],
                 empty_label=17,
                 voxel_size=[],
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 is_pretrain=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_pts=dict(type='L1Loss'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.num_fu_query = num_fu_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.num_fu_frames = num_fu_frames
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.empty_label = empty_label
        self.loss_cls = build_loss(loss_cls)
        self.loss_pts = build_loss(loss_pts)
        self.loss_stamp = torch.nn.BCEWithLogitsLoss()
        self.transformer = build_transformer(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims
        self.voxel_generator = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points=10,
            max_voxels=self.num_query * self.num_refines[-1],
            deterministic=False
        )

        # prepare scene
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()
        self.pretrain = is_pretrain
        self.freeze_tass = False
        self.ind_stamps_all = None
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)

        self._init_layers()
        self.register_buffer(
            'scene_size_new', torch.tensor([160, 120, 9.6]),
            persistent=False)
        self.register_buffer(
            'pc_range_new', torch.tensor([-80, -60, -3, 80, 60, 6.6]),
            persistent=False)
        self.register_buffer(
            'voxel_num_new', torch.tensor([400, 300, 24]),
            persistent=False)
        self.register_buffer(
            'num_stamps_all',
            torch.ones(
                self.num_query + sum(self.num_fu_query),
                self.num_fu_frames + 1).long())


    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query + sum(self.num_fu_query), 3)
        nn.init.uniform_(self.init_points.weight[:,0], 0, 1.1)
        nn.init.uniform_(self.init_points.weight[:,1:],0,1)

    def init_weights(self):
        self.transformer.init_weights()

    def forward(self, mlvl_feats, img_metas):

        B, Q = mlvl_feats[0].shape[0], self.num_query + sum(self.num_fu_query)
        # self.init_points.weight.data.clamp_(0.0, 1.0)
        init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1)

        if getattr(self,'points_scale',None) is not None:
            init_points = init_points * self.points_scale.unsqueeze(1)
            # from open3d_vis_utils import draw_scenes
            # draw_scenes(init_points.reshape(-1,3))
        query_feat = init_points.new_zeros(B, Q, self.embed_dims)
        if getattr(self,'ind_stamps_all',None) is None:
            num_stamps_all = self.num_stamps_all.float()
            num_stamps_all = num_stamps_all / torch.sum(num_stamps_all, -1, keepdim=True)
            ind_stamps_all = get_matched_inds(num_stamps_all, [self.num_query] + self.num_fu_query)
            self.ind_stamps_all = ind_stamps_all
            if not self.training or getattr(self, 'freeze_tass', False):
                self.reset_mask()

        # fu_query_feat = fu_init_points.new_zeros(B, fu_Q, self.embed_dims)
        if self.pretrain and not getattr(self, 'freeze_tass', False):
            num_stamps_all = self.num_stamps_all.float()
            num_stamps_all = num_stamps_all / torch.sum(num_stamps_all, -1, keepdim=True)
            ind_stamps_all = get_matched_inds(num_stamps_all, [self.num_query] + self.num_fu_query)
            self.ind_stamps_all = ind_stamps_all

        query_feat, cls_scores, refine_pts = self.transformer(
            init_points,
            query_feat,
            mlvl_feats,
            img_metas=img_metas,

        )

        return dict(init_points=init_points,
                    all_cls_scores=cls_scores,
                    all_refine_pts=refine_pts,
                    query_feat=query_feat,
                    )

    def get_dis_weight(self, pts):
        max_dist = torch.sqrt(
            self.scene_size[0] ** 2 + self.scene_size[1] ** 2)
        centers = (self.pc_range[:3] + self.pc_range[3:]) / 2
        dist = (pts - centers[None, ...])[..., :2]
        dist = torch.norm(dist, dim=-1)
        return dist / max_dist + 1

    def discretize(self, pts, clip=True, decode=False):
        loc = torch.floor((pts - self.pc_range[:3]) / self.voxel_size)
        if clip:
            loc[..., 0] = loc[..., 0].clamp(0, self.voxel_num[0] - 1)
            loc[..., 1] = loc[..., 1].clamp(0, self.voxel_num[1] - 1)
            loc[..., 2] = loc[..., 2].clamp(0, self.voxel_num[2] - 1)

        return loc.long() if not decode else \
            (loc + 0.5) * self.voxel_size + self.pc_range[:3]
    def reset_mask(self):
        init_points = self.init_points.weight.data
        ind_mask = init_points.new_zeros(init_points.shape[0], init_points.shape[0])
        for i in range(self.num_fu_frames):
            row_idx = (self.ind_stamps_all == i).nonzero(as_tuple=True)[0]
            col_idx = (self.ind_stamps_all > i).nonzero(as_tuple=True)[0]
            grid_row, grid_col = torch.meshgrid(row_idx, col_idx, indexing='ij')
            ind_mask[grid_row, grid_col] = -1e5
        for decoder_layer in self.transformer.decoder.decoder_layers:
            decoder_layer.self_attn.ind_mask = ind_mask
    @torch.no_grad()
    def _get_target_single(self, refine_pts, gt_points, gt_labels):
        # knn to apply Chamfer distance
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).reshape(-1).long()

        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])

        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).reshape(-1).long()
        gt_paired_pts = refine_pts[gt_paired_idx]
        pred_paired_pts = gt_points[pred_paired_idx]

        # cls assignment
        refine_pts_labels = gt_labels[pred_paired_idx]
        train_cfg = self.train_cfg or {}
        cls_weights = train_cfg.get('cls_weights', [1] * self.num_classes)
        cls_weights = refine_pts.new_tensor(cls_weights)
        label_weights = cls_weights * \
                        self.get_dis_weight(pred_paired_pts)[..., None]

        # gt side assignment
        empty_dist_thr = train_cfg.get('empty_dist_thr', 0.2)
        empty_weights = train_cfg.get('empty_weights', 3)

        gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        dist = torch.norm(gt_points - gt_paired_pts, dim=-1)
        mask = (dist > empty_dist_thr)
        gt_pts_weights[mask] = empty_weights
        if True:
            dis = torch.norm(pred_paired_pts - refine_pts,dim=-1,p=2,keepdim=True)
            label_weights = label_weights * torch.clamp(1/dis,max=1,min=0.4)
            gt_pts_weights[(gt_labels>=15) | (gt_labels==11)] *=0.5


        return (refine_pts_labels, gt_paired_idx, pred_paired_idx, label_weights,
                gt_pts_weights)
    def gather(self,tensor,sampled_inds):
        B,N = sampled_inds.shape[:2]
        batch_indx = torch.arange(tensor.shape[0],device=sampled_inds.device).unsqueeze(1).repeat(1,N)
        return tensor[batch_indx,sampled_inds]

    def get_targets(self):
        # To instantiate the abstract method
        pass

    def loss_single_mask(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_labels_list,
                    gt_stamps_list,
                    temporal_reweight=False):

        num_imgs,num_query,num_pts = cls_scores.shape[:3]  # B
        cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)

        if temporal_reweight:
            for i in range(num_imgs):
                gt_points = gt_points_list[i]
                gt_stamps = gt_stamps_list[i]
                mask = gt_stamps[:,0]==1
                mask[gt_points[:,0]>40] = True
                mask[(gt_points[:,0]>35) * (gt_stamps[:,2]==1)] = True
                mask[(gt_points[:,0]>30) * (gt_stamps[:,1] == 1)] = True
                gt_points_list[i] = gt_points[mask]
                gt_stamps_list[i] = gt_stamps[mask]
                gt_labels_list[i] = gt_labels_list[i][mask]

        refine_pts = decode_points(refine_pts, self.pc_range)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights,
         gt_pts_weights) = multi_apply(
            self._get_target_single, refine_pts_list, gt_points_list,
             gt_labels_list)

        gt_paired_pts, pred_paired_pts, gt_weights,pred_paired_stamps = [], [], [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])

            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])
            pred_paired_stamps.append(gt_stamps_list[i][pred_paired_idx_list[i]])

            fore_mask = (labels_list[i].reshape(num_query,-1,1)<2) | (labels_list[i].reshape(num_query,-1,1)>10)
            if not getattr(self, 'freeze_tass', False):
                self.num_stamps_all += (gt_stamps_list[i][pred_paired_idx_list[i]].reshape(num_query,num_pts,self.num_fu_frames+1) * fore_mask).sum(1)

            # gt_pts_weights[i] = gt_pts_weights[i] * mask[gt_paired_idx_list[i]].squeeze(-1) * dist_mask[i]


        # concatenate all results from different samples
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        cls_weights = torch.cat(cls_weights)
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)


        # cls_weights = cls_weights * mask
        # calculate loss cls
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 weight=cls_weights,
                                 avg_factor=cls_scores.shape[0])
        # calculate loss pts

        loss_pts = self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])

        if False:
            dis = torch.norm(pred_pts-pred_paired_pts,dim=-1,p=2,keepdim=True)
            pred_weights = torch.clamp(dis,max=1.5,min=0.4)

        loss_pts += self.loss_pts(pred_pts,
                                  pred_paired_pts,
                                  avg_factor=pred_pts.shape[0])


        # loss_stamp = F.binary_cross_entropy_with_logits(pred_stamps[moving_mask],
        #                              pred_paired_stamps[moving_mask].float(),
        #                              reduction='none',
        #                              ).sum() / (moving_mask.sum() * (self.num_fu_frames+1))
        return loss_cls, loss_pts



    def loss_single(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_labels_list):
        num_imgs = cls_scores.size(0)  # B
        cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights,
         gt_pts_weights) = multi_apply(
            self._get_target_single, refine_pts_list, gt_points_list,
             gt_labels_list)

        gt_paired_pts, pred_paired_pts = [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        # concatenate all results from different samples
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        cls_weights = torch.cat(cls_weights)
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)


        # calculate loss cls
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 weight=cls_weights,
                                 avg_factor=cls_scores.shape[0])
        # calculate loss pts
        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts,
                                  pred_paired_pts,
                                  avg_factor=pred_pts.shape[0])

        return loss_cls, loss_pts

    def loss_single_rangemask(self,
                    cls_scores,
                    refine_pts,
                    points_mask,
                    gt_points_list,
                    gt_labels_list):
        num_imgs = cls_scores.size(0)  # B

        cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        if points_mask is None:
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
            refine_pts_list = [refine_pts[i] for i in range(num_imgs)]
        else:
            points_mask = points_mask.reshape(num_imgs, -1)
            points_mask = points_mask.reshape(-1,1)
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
            refine_pts_list = [refine_pts[i] for i in range(num_imgs)]
        (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights,
         gt_pts_weights) = multi_apply(
            self._get_target_single, refine_pts_list, gt_points_list,
             gt_labels_list)

        gt_paired_pts, pred_paired_pts = [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        # concatenate all results from different samples
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        cls_weights = torch.cat(cls_weights)
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)

        # calculate loss cls
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 weight=cls_weights *points_mask,
                                 avg_factor=cls_scores.shape[0])
        # calculate loss pts
        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts,
                                  pred_paired_pts,
                                  weight=points_mask,
                                  avg_factor=pred_pts.shape[0])

        return loss_cls, loss_pts



    @staticmethod
    def _labels_in_classes(labels, class_ids):
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in class_ids:
            mask |= labels == class_id
        return mask

    @staticmethod
    def _assign_motion_state_roles(gt_points, gt_labels, role_metadata,
                                   static_ids):
        """Assign continuous motion roles with conservative validity masks."""
        role_target = gt_points.new_zeros(gt_points.shape[0])
        role_valid = torch.zeros(
            gt_points.shape[0], device=gt_points.device, dtype=torch.bool)
        static_mask = OPUSHead._labels_in_classes(gt_labels, static_ids)
        role_valid[static_mask] = True
        if role_metadata is None or gt_points.numel() == 0:
            return role_target, role_valid
        centers = role_metadata['centers'].to(
            device=gt_points.device, dtype=gt_points.dtype)
        radius = role_metadata['radius'].to(
            device=gt_points.device, dtype=gt_points.dtype)
        actor_role = role_metadata['role'].to(
            device=gt_points.device, dtype=gt_points.dtype)
        actor_valid = role_metadata.get(
            'valid', torch.ones_like(actor_role, dtype=torch.bool)).to(
                device=gt_points.device).bool()
        if centers.numel() == 0:
            return role_target, role_valid
        dims = role_metadata.get('dims')
        yaw = role_metadata.get('yaw')
        if dims is not None and yaw is not None and dims.numel() > 0:
            dims = dims.to(device=gt_points.device, dtype=gt_points.dtype)
            yaw = yaw.to(device=gt_points.device, dtype=gt_points.dtype)
            delta = gt_points[:, None, :3] - centers[None, :, :3]
            cos_yaw = yaw.cos()[None, :]
            sin_yaw = yaw.sin()[None, :]
            # Rotate points into each actor's local box frame.
            local_x = cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1]
            local_y = -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1]
            inside = (local_x.abs() <= dims[None, :, 0] * 0.5) & \
                (local_y.abs() <= dims[None, :, 1] * 0.5)
            if gt_points.shape[-1] >= 3 and dims.shape[-1] >= 3:
                inside = inside & (
                    delta[..., 2].abs() <= dims[None, :, 2] * 0.5)
            # Select the nearest actor among boxes that actually contain the
            # point.  This avoids assigning the rounded box corners or a
            # distant overlapping actor to a voxel.
            center_distance = torch.cdist(
                gt_points[:, :2], centers[:, :2], p=2)
            center_distance = center_distance.masked_fill(~inside, float('inf'))
            nearest_distance, nearest_actor = center_distance.min(dim=1)
            matched = torch.isfinite(nearest_distance) & \
                actor_valid[nearest_actor] & ~static_mask
        else:
            distances = torch.cdist(gt_points[:, :2], centers[:, :2], p=2)
            nearest_distance, nearest_actor = distances.min(dim=1)
            matched = (nearest_distance <= radius[nearest_actor]) & \
                actor_valid[nearest_actor] & ~static_mask
        role_target[matched] = actor_role[nearest_actor[matched]]
        role_valid[matched] = True
        return role_target, role_valid

    @torch.no_grad()
    def build_future_match_cache(self, refine_pts, voxel_semantics,
                                 role_metadata=None):
        """Build one reusable bidirectional KNN assignment for a future step."""
        batch_size, num_query, num_points = refine_pts.shape[:3]
        decoded_points = decode_points(
            refine_pts.reshape(batch_size, -1, 3), self.pc_range)
        gt_points_list, gt_labels_list = self.get_sparse_voxels(
            voxel_semantics)

        labels_list = []
        gt_paired_idx_list = []
        pred_paired_idx_list = []
        label_weights_list = []
        gt_pts_weights_list = []
        valid_gt_list = []
        gt_role_target_list = []
        gt_role_valid_list = []
        for batch_index in range(batch_size):
            gt_points = gt_points_list[batch_index]
            valid_gt = gt_points.shape[0] > 0
            valid_gt_list.append(valid_gt)
            if valid_gt:
                match = self._get_target_single(
                    decoded_points[batch_index],
                    gt_points,
                    gt_labels_list[batch_index])
                labels, gt_index, pred_index, label_weight, gt_weight = match
            else:
                num_predictions = decoded_points.shape[1]
                labels = torch.full(
                    (num_predictions,), self.num_classes,
                    device=decoded_points.device, dtype=torch.long)
                gt_index = torch.empty(
                    0, device=decoded_points.device, dtype=torch.long)
                pred_index = torch.empty(
                    0, device=decoded_points.device, dtype=torch.long)
                label_weight = decoded_points.new_ones(
                    num_predictions, self.num_classes)
                gt_weight = decoded_points.new_empty(0)
            static_ids = getattr(self, 'dsqe_cfg', {}).get(
                'static_class_ids', [1, 8, 11, 12, 13, 14, 15, 16])
            motion = None if role_metadata is None or batch_index >= len(role_metadata) \
                else role_metadata[batch_index]
            gt_role_target, gt_role_valid = self._assign_motion_state_roles(
                gt_points, gt_labels_list[batch_index], motion, static_ids)
            labels_list.append(labels)
            gt_paired_idx_list.append(gt_index)
            pred_paired_idx_list.append(pred_index)
            label_weights_list.append(label_weight)
            gt_pts_weights_list.append(gt_weight)
            gt_role_target_list.append(gt_role_target)
            gt_role_valid_list.append(gt_role_valid)

        config = getattr(self, 'dsqe_cfg', {})
        labels = torch.stack(labels_list).reshape(
            batch_size, num_query, num_points)
        role_target_list = []
        role_valid_list = []
        for batch_index in range(batch_size):
            # ``pred_paired_idx`` maps every predicted point to its nearest
            # GT point (the KNN helper's second argument is the query set).
            point_to_gt = pred_paired_idx_list[batch_index]
            gt_role_target = gt_role_target_list[batch_index]
            gt_role_valid = gt_role_valid_list[batch_index]
            if point_to_gt.numel() == 0:
                role_target_list.append(labels[batch_index].new_zeros(
                    labels[batch_index].shape, dtype=torch.float32))
                role_valid_list.append(torch.zeros_like(
                    labels[batch_index], dtype=torch.bool))
            else:
                role_target_list.append(gt_role_target[point_to_gt].float())
                role_valid_list.append(gt_role_valid[point_to_gt])
        role_target = torch.stack(role_target_list).reshape(
            batch_size, num_query, num_points)
        role_valid = torch.stack(role_valid_list).reshape(
            batch_size, num_query, num_points)
        return dict(
            labels_list=labels_list,
            gt_paired_idx_list=gt_paired_idx_list,
            pred_paired_idx_list=pred_paired_idx_list,
            label_weights_list=label_weights_list,
            gt_pts_weights_list=gt_pts_weights_list,
            valid_gt_list=valid_gt_list,
            gt_points_list=gt_points_list,
            gt_labels_list=gt_labels_list,
            gt_role_target_list=gt_role_target_list,
            gt_role_valid_list=gt_role_valid_list,
            role_target=role_target,
            role_valid=role_valid,
        )

    def _loss_future_cached(self, cls_scores, refine_pts, points_mask, cache):
        batch_size = cls_scores.shape[0]
        cls_scores_flat = cls_scores.reshape(
            batch_size, -1, self.num_classes)
        refine_pts_flat = decode_points(
            refine_pts.reshape(batch_size, -1, 3), self.pc_range)
        if points_mask is None:
            points_mask = refine_pts_flat.new_ones(
                refine_pts_flat.shape[:2])
        else:
            points_mask = points_mask.reshape(batch_size, -1).to(
                refine_pts_flat.dtype)

        gt_paired_points = []
        pred_points = []
        pred_paired_points = []
        pred_range_weights = []
        valid_gt_list = cache.get(
            'valid_gt_list', [True] * batch_size)
        for batch_index in range(batch_size):
            gt_paired_points.append(refine_pts_flat[batch_index][
                cache['gt_paired_idx_list'][batch_index]])
            if valid_gt_list[batch_index]:
                pred_points.append(refine_pts_flat[batch_index])
                pred_paired_points.append(
                    cache['gt_points_list'][batch_index][
                        cache['pred_paired_idx_list'][batch_index]])
                pred_range_weights.append(points_mask[batch_index])

        scores = torch.cat([score for score in cls_scores_flat])
        labels = torch.cat(cache['labels_list'])
        label_weights = torch.cat(cache['label_weights_list'])
        range_weights = points_mask.reshape(-1, 1)
        loss_cls = self.loss_cls(
            scores,
            labels,
            weight=label_weights * range_weights,
            avg_factor=max(scores.shape[0], 1))

        zero = refine_pts_flat.sum() * 0
        gt_points = torch.cat(cache['gt_points_list'])
        gt_paired_points = torch.cat(gt_paired_points)
        gt_weights = torch.cat(cache['gt_pts_weights_list'])
        loss_pts = zero
        if gt_points.shape[0] > 0:
            loss_pts = self.loss_pts(
                gt_points,
                gt_paired_points,
                weight=gt_weights[..., None],
                avg_factor=gt_points.shape[0])
        if pred_points:
            pred_points = torch.cat(pred_points)
            pred_paired_points = torch.cat(pred_paired_points)
            pred_range_weights = torch.cat(pred_range_weights).reshape(-1, 1)
            loss_pts = loss_pts + self.loss_pts(
                pred_points,
                pred_paired_points,
                weight=pred_range_weights,
                avg_factor=pred_points.shape[0])
        return loss_cls, loss_pts, refine_pts_flat

    @staticmethod
    def _masked_mean(values, mask):
        mask = mask.to(values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _loss_role(self, output, cache):
        role_logits = output['role_logits'].squeeze(-1)
        role_target = cache['role_target'].to(role_logits.dtype)
        role_valid = cache['role_valid']
        config = self.dsqe_cfg
        gamma = config.get('role_gamma', 2.0)
        alpha = config.get('role_alpha', 0.75)

        bce = F.binary_cross_entropy_with_logits(
            role_logits, role_target, reduction='none')
        role_probability = role_logits.sigmoid()
        pt = role_probability * role_target + \
            (1 - role_probability) * (1 - role_target)
        alpha_weight = alpha * role_target + \
            (1 - alpha) * (1 - role_target)
        point_loss = self._masked_mean(
            alpha_weight * (1 - pt).pow(gamma) * bce, role_valid)

        pool_weights = output['pool_weights'].squeeze(-1)
        valid_weights = pool_weights * role_valid.to(pool_weights.dtype)
        query_target = (valid_weights * role_target).sum(dim=2) / \
            valid_weights.sum(dim=2).clamp_min(1e-6)
        query_valid = role_valid.any(dim=2)
        query_probability = output['query_role'].squeeze(-1).clamp(
            1e-5, 1 - 1e-5)
        query_bce = F.binary_cross_entropy(
            query_probability, query_target, reduction='none')
        query_loss = self._masked_mean(query_bce, query_valid)
        return point_loss + config.get('role_query_weight', 0.5) * query_loss

    def _loss_static(self, output, cache):
        reference = output.get('static_reference_metric')
        if reference is None or output['num_carried'] == 0:
            return output['points_metric'].sum() * 0
        num_carried = output['num_carried']
        role_target = cache['role_target'][:, :num_carried]
        role_valid = cache['role_valid'][:, :num_carried]
        static_mask = role_valid & (role_target < 0.5)
        error = (output['points_metric'][:, :num_carried] - reference).abs()
        return self._masked_mean(error, static_mask)

    def _loss_dynamic(self, cls_scores, output, cache, refine_pts_flat):
        dynamic_ids = self.dsqe_cfg.get(
            'dynamic_class_ids', [2, 3, 4, 5, 6, 7, 9, 10])
        batch_size = cls_scores.shape[0]
        scores_flat = cls_scores.reshape(
            batch_size, -1, self.num_classes)
        pred_distance_sum = refine_pts_flat.new_tensor(0.0)
        gt_distance_sum = refine_pts_flat.new_tensor(0.0)
        pred_count = refine_pts_flat.new_tensor(0.0)
        gt_count = refine_pts_flat.new_tensor(0.0)
        dynamic_scores = []
        dynamic_labels = []
        dynamic_weights = []

        valid_gt_list = cache.get('valid_gt_list', [True] * batch_size)
        for batch_index in range(batch_size):
            if not valid_gt_list[batch_index]:
                continue
            gt_points = cache['gt_points_list'][batch_index]
            point_to_gt = cache['pred_paired_idx_list'][batch_index]
            pred_target = gt_points[point_to_gt]
            count = min(refine_pts_flat.shape[1], pred_target.shape[0])
            pred_error = refine_pts_flat.new_zeros(refine_pts_flat.shape[1])
            if count > 0:
                pred_error[:count] = (
                    refine_pts_flat[batch_index, :count] -
                    pred_target[:count]).abs().mean(dim=-1)
            role_target = cache['role_target'][batch_index].reshape(-1)
            role_valid = cache['role_valid'][batch_index].reshape(-1)
            pred_role = output['role_pred'][batch_index].reshape(-1)
            # Softly weight every valid candidate.  There is intentionally no
            # ``pred_role > 0.5`` branch, so low-confidence roles still train.
            # Keep a non-zero floor for low-confidence roles and normalize by
            # GT role mass rather than the same prediction-dependent weight.
            # This preserves a useful geometric gradient even when every
            # predicted role is below 0.5.
            dynamic_weight = role_target * (
                0.25 + 0.75 * pred_role) * \
                role_valid.to(pred_role.dtype)
            dynamic_weight = dynamic_weight[:pred_error.shape[0]]
            pred_distance_sum = pred_distance_sum + \
                (pred_error * dynamic_weight).sum()
            pred_count = pred_count + (
                role_target[:pred_error.shape[0]] *
                role_valid[:pred_error.shape[0]].to(pred_role.dtype)).sum()

            gt_role_targets = cache.get('gt_role_target_list')
            gt_role_valids = cache.get('gt_role_valid_list')
            gt_role_target = None if gt_role_targets is None or \
                batch_index >= len(gt_role_targets) else \
                gt_role_targets[batch_index]
            gt_role_valid = None if gt_role_valids is None or \
                batch_index >= len(gt_role_valids) else \
                gt_role_valids[batch_index]
            if gt_role_target is not None and gt_role_target.numel() > 0:
                # Keep every non-static actor candidate and use its soft
                # motion role as a continuous geometry weight.
                dynamic_gt_mask = gt_role_valid & (gt_role_target > 0)
                if dynamic_gt_mask.any():
                    dynamic_gt_points = cache['gt_points_list'][batch_index][
                        dynamic_gt_mask]
                    dynamic_gt_role = gt_role_target[dynamic_gt_mask]
                    pred_points = refine_pts_flat[batch_index]
                    # GT -> prediction coverage: every moving GT point must
                    # find a predicted point.  The previous implementation
                    # repeated the prediction -> GT direction and could
                    # therefore miss uncovered moving regions.
                    nearest_pred = self._chunked_nearest_indices(
                        dynamic_gt_points, pred_points,
                        self.dsqe_cfg.get('dynamic_cdist_chunk_size', 1024))
                    gt_error = (dynamic_gt_points -
                                pred_points[nearest_pred]).abs().mean(dim=-1)
                    gt_weight = dynamic_gt_role * (
                        0.25 + 0.75 * pred_role[nearest_pred])
                    gt_distance_sum = gt_distance_sum + \
                        (gt_weight * gt_error).sum()
                    gt_count = gt_count + dynamic_gt_role.sum()

            labels = cache['labels_list'][batch_index]
            semantic_dynamic_mask = self._labels_in_classes(labels, dynamic_ids)
            if semantic_dynamic_mask.any():
                dynamic_scores.append(scores_flat[batch_index][semantic_dynamic_mask])
                dynamic_labels.append(labels[semantic_dynamic_mask])
                dynamic_weights.append(
                    cache['label_weights_list'][batch_index][
                        semantic_dynamic_mask])

        distance_loss = pred_distance_sum / pred_count.clamp_min(1) + \
            gt_distance_sum / gt_count.clamp_min(1)
        classification_loss = refine_pts_flat.sum() * 0
        if dynamic_scores:
            dynamic_scores = torch.cat(dynamic_scores)
            dynamic_labels = torch.cat(dynamic_labels)
            dynamic_weights = torch.cat(dynamic_weights)
            classification_loss = self.loss_cls(
                dynamic_scores,
                dynamic_labels,
                weight=dynamic_weights,
                avg_factor=max(dynamic_scores.shape[0], 1))
        return distance_loss + self.dsqe_cfg.get(
            'dynamic_cls_weight', 1.0) * classification_loss

    @staticmethod
    def _chunked_nearest_indices(query_points, reference_points,
                                 chunk_size=1024):
        """Exact L1 nearest neighbors with bounded pairwise memory."""
        if query_points.shape[0] == 0 or reference_points.shape[0] == 0:
            return torch.empty(
                query_points.shape[0], device=query_points.device,
                dtype=torch.long)
        if chunk_size <= 0:
            raise ValueError('dynamic_cdist_chunk_size must be positive')

        nearest_indices = []
        with torch.no_grad():
            query = query_points.detach().float()
            reference = reference_points.detach().float()
            for query_start in range(0, query.shape[0], chunk_size):
                query_chunk = query[query_start:query_start + chunk_size]
                best_distance = query_chunk.new_full(
                    (query_chunk.shape[0],), float('inf'))
                best_index = torch.zeros(
                    query_chunk.shape[0], device=query.device,
                    dtype=torch.long)
                for ref_start in range(0, reference.shape[0], chunk_size):
                    reference_chunk = reference[
                        ref_start:ref_start + chunk_size]
                    distance = torch.cdist(
                        query_chunk, reference_chunk, p=1)
                    chunk_distance, chunk_index = distance.min(dim=1)
                    update = chunk_distance < best_distance
                    best_distance = torch.where(
                        update, chunk_distance, best_distance)
                    best_index = torch.where(
                        update, chunk_index + ref_start, best_index)
                nearest_indices.append(best_index)
        return torch.cat(nearest_indices)

    def _loss_ego(self, output):
        target = output.get('gt_relative_pose')
        if target is None:
            return output['predicted_relative_pose'].sum() * 0
        prediction = output['predicted_relative_pose']
        translation_loss = F.l1_loss(
            prediction[..., :2], target[..., :2])
        prediction_yaw = F.normalize(prediction[..., 2:4], dim=-1)
        target_yaw = F.normalize(target[..., 2:4], dim=-1)
        yaw_loss = 1 - (prediction_yaw * target_yaw).sum(dim=-1).mean()
        return translation_loss + self.dsqe_cfg.get(
            'ego_yaw_weight', 1.0) * yaw_loss

    def _loss_leak(self, output, cache):
        if output['num_carried'] == 0:
            return output['query_motion'].sum() * 0
        num_carried = output['num_carried']
        role_target = cache['role_target'][:, :num_carried]
        role_valid = cache['role_valid'][:, :num_carried]
        static_fraction = (role_valid & (role_target < 0.5)).float().mean(dim=2)
        motion = output['query_motion'].abs().mean(dim=-1)
        return (motion * static_fraction).sum() / \
            static_fraction.sum().clamp_min(1.0)

    @staticmethod
    def _loss_smooth(previous_output, output):
        if previous_output is None:
            return output['query_motion'].sum() * 0
        overlap = min(previous_output['query_motion'].shape[1],
                      output['query_motion'].shape[1])
        if overlap == 0:
            return output['query_motion'].sum() * 0
        return F.l1_loss(
            output['query_motion'][:, :overlap],
            previous_output['query_motion'][:, :overlap])

    @staticmethod
    def _role_diagnostics(output, cache):
        target = cache['role_target'] >= 0.5
        valid = cache['role_valid']
        prediction = output['role_pred'].squeeze(-1) >= 0.5
        correct = ((prediction == target) & valid).sum()
        accuracy = correct.to(output['role_pred'].dtype) / \
            valid.sum().clamp_min(1)
        prediction_float = output['role_pred'].squeeze(-1)
        valid_float = valid.to(prediction_float.dtype)
        dynamic_ratio = (prediction_float * valid_float).sum() / \
            valid_float.sum().clamp_min(1)
        gt_dynamic = (target & valid)
        pred_dynamic = prediction & valid
        true_positive = (pred_dynamic & gt_dynamic).sum().float()
        precision = true_positive / pred_dynamic.sum().clamp_min(1).float()
        recall = true_positive / gt_dynamic.sum().clamp_min(1).float()
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
        moving_mean = prediction_float[gt_dynamic].mean() \
            if gt_dynamic.any() else prediction_float.sum() * 0
        static_mask = valid & ~target
        static_mean = prediction_float[static_mask].mean() \
            if static_mask.any() else prediction_float.sum() * 0
        low = ((prediction_float < 0.05) & valid).float().sum() / \
            valid_float.sum().clamp_min(1)
        high = ((prediction_float > 0.95) & valid).float().sum() / \
            valid_float.sum().clamp_min(1)
        dynamic_from_static, static_from_dynamic = output['interaction_gates']
        return dict(
            role_accuracy=accuracy.detach(),
            dynamic_ratio=dynamic_ratio.detach(),
            dynamic_precision=precision.detach(),
            dynamic_recall=recall.detach(),
            dynamic_f1=f1.detach(),
            gt_dynamic_ratio=gt_dynamic.float().sum().detach() /
            valid_float.sum().clamp_min(1).detach(),
            pred_dynamic_ratio=pred_dynamic.float().sum().detach() /
            valid_float.sum().clamp_min(1).detach(),
            mean_role_on_moving_gt=moving_mean.detach(),
            mean_role_on_static_gt=static_mean.detach(),
            role_saturation_low=low.detach(),
            role_saturation_high=high.detach(),
            gate_dynamic_from_static=dynamic_from_static.detach(),
            gate_static_from_dynamic=static_from_dynamic.detach())

    @staticmethod
    def _residual_diagnostics(output):
        values = {}
        for name in ('dynamic_delta', 'static_delta'):
            residual = output.get(name)
            if residual is None:
                continue
            values[name + '_mean_x'] = residual[..., 0].mean().detach()
            values[name + '_mean_y'] = residual[..., 1].mean().detach()
            values[name + '_mean_z'] = residual[..., 2].mean().detach()
            values[name + '_abs_mean_x'] = residual[..., 0].abs().mean().detach()
            values[name + '_abs_mean_y'] = residual[..., 1].abs().mean().detach()
            values[name + '_abs_mean_z'] = residual[..., 2].abs().mean().detach()
            flat = residual.reshape(-1, 3).abs()
            p95 = torch.quantile(flat, 0.95, dim=0)
            values[name + '_p95_x'] = p95[0].detach()
            values[name + '_p95_y'] = p95[1].detach()
            values[name + '_p95_z'] = p95[2].detach()
        return values

    @staticmethod
    def _motion_diagnostics(output):
        motion = output.get('query_motion')
        if motion is None or motion.numel() == 0:
            return {}
        values = {}
        for axis, label in enumerate(('x', 'y', 'z')):
            values['query_motion_mean_' + label] = \
                motion[..., axis].mean().detach()
        p95 = torch.quantile(
            motion.reshape(-1, 3).abs(), 0.95, dim=0)
        for axis, label in enumerate(('x', 'y', 'z')):
            values['query_motion_p95_' + label] = p95[axis].detach()
        return values

    def _geometry_diagnostics(self, output, cache):
        """Report soft static-warp and dynamic-displacement errors."""
        values = {}
        num_carried = output.get('num_carried', 0)
        role_target = cache.get('role_target')
        role_valid = cache.get('role_valid')
        if (num_carried > 0 and role_target is not None and
                role_valid is not None and
                output.get('static_reference_metric') is not None):
            reference = output['static_reference_metric']
            points = output['points_metric'][:, :num_carried]
            static_error = (points - reference).abs()
            static_mask = role_valid[:, :num_carried] & \
                (role_target[:, :num_carried] < 0.5)
            denom = static_mask.to(static_error.dtype).sum().clamp_min(1.0)
            for axis, label in enumerate(('x', 'y', 'z')):
                values['static_warp_error_' + label] = \
                    (static_error[..., axis] * static_mask.to(
                        static_error.dtype)).sum().detach() / denom

        gt_roles = cache.get('gt_role_target_list')
        gt_valids = cache.get('gt_role_valid_list')
        if gt_roles is None or gt_valids is None:
            return values
        dynamic_errors = []
        for batch_index, (gt_role, gt_valid) in enumerate(
                zip(gt_roles, gt_valids)):
            dynamic_mask = gt_valid & (gt_role > 0)
            if not dynamic_mask.any():
                continue
            gt_points = cache['gt_points_list'][batch_index][dynamic_mask]
            gt_role = gt_role[dynamic_mask]
            batch_points = output['points_metric'][batch_index].reshape(-1, 3)
            nearest_pred = OPUSHead._chunked_nearest_indices(
                gt_points.detach(), batch_points.detach(),
                getattr(self, 'dsqe_cfg', {}).get(
                    'dynamic_cdist_chunk_size', 1024)
                if hasattr(self, 'dsqe_cfg') else 1024)
            matched_points = batch_points[nearest_pred]
            matched_roles = output['role_pred'][batch_index].reshape(
                -1).detach()[nearest_pred]
            dynamic_errors.append(((matched_points - gt_points).abs(),
                                   (0.25 + 0.75 * matched_roles) *
                                   gt_role.detach()))
        if dynamic_errors:
            dynamic_error = torch.cat([value for value, _ in dynamic_errors],
                                      dim=0)
            dynamic_weight = torch.cat([weight for _, weight in dynamic_errors],
                                        dim=0)
            denom = dynamic_weight.sum().clamp_min(1e-6)
            for axis, label in enumerate(('x', 'y', 'z')):
                values['dynamic_displacement_error_' + label] = \
                    (dynamic_error[..., axis] * dynamic_weight).sum().detach() \
                    / denom
        return values

    def loss_future(self,
                    voxel_semantics,
                    all_refine_pts,
                    all_cls_scores,
                    points_mask,
                    dsqe_outputs=None,
                    match_cache_list=None):
        loss_dict = dict()
        previous_output = None
        config = getattr(self, 'dsqe_cfg', {})
        for index, (voxel_semantic, refine_pts, cls_scores) in enumerate(zip(
                voxel_semantics, all_refine_pts, all_cls_scores)):
            prefix = 'fu{}'.format(index + 1)
            if self.pretrain:
                output = None if dsqe_outputs is None else dsqe_outputs[index]
                residual_graph = refine_pts.sum() + cls_scores.sum()
                if output is not None:
                    for name in ('delta_feat', 'delta_points',
                                 'semantic_correction', 'query_motion'):
                        value = output.get(name)
                        if value is not None:
                            residual_graph = residual_graph + value.sum()
                zero = residual_graph * 0
                loss_dict[prefix + '.loss_cls'] = zero
                loss_dict[prefix + '.loss_pts'] = zero
                if dsqe_outputs is not None:
                    cache = None if match_cache_list is None else \
                        match_cache_list[index]
                    if cache is not None:
                        role_loss = self._loss_role(output, cache)
                        ego_loss = self._loss_ego(output)
                        loss_dict[prefix + '.loss_role'] = \
                            config.get('lambda_role', 0.5) * role_loss
                        loss_dict[prefix + '.loss_ego'] = \
                            config.get('lambda_ego', 1.0) * ego_loss
                        for name, value in self._role_diagnostics(
                                output, cache).items():
                            loss_dict[prefix + '.' + name] = value
                        for name, value in self._residual_diagnostics(output).items():
                            loss_dict[prefix + '.' + name] = value
                        for name, value in self._motion_diagnostics(output).items():
                            loss_dict[prefix + '.' + name] = value
                        for name, value in self._geometry_diagnostics(
                                output, cache).items():
                            loss_dict[prefix + '.' + name] = value
                    else:
                        role_zero = output['role_logits'].sum() * 0
                        loss_dict[prefix + '.loss_role'] = role_zero
                        loss_dict[prefix + '.loss_ego'] = zero
                    for name in ('static', 'dynamic', 'smooth', 'leak'):
                        loss_dict[prefix + '.loss_' + name] = zero
                continue

            cache = None if match_cache_list is None else \
                match_cache_list[index]
            if cache is None:
                cache = self.build_future_match_cache(
                    refine_pts, voxel_semantic)
            point_mask = None if points_mask is None else points_mask[index]
            loss_cls, loss_pts, refine_pts_flat = self._loss_future_cached(
                cls_scores, refine_pts, point_mask, cache)
            loss_dict[prefix + '.loss_cls'] = loss_cls
            loss_dict[prefix + '.loss_pts'] = loss_pts

            if dsqe_outputs is not None:
                output = dsqe_outputs[index]
                role_loss = self._loss_role(output, cache)
                ego_loss = self._loss_ego(output)
                loss_dict[prefix + '.loss_role'] = \
                    config.get('lambda_role', 0.5) * role_loss
                loss_dict[prefix + '.loss_ego'] = \
                    config.get('lambda_ego', 1.0) * ego_loss

                geometry_weight = float(not self.pretrain)
                loss_dict[prefix + '.loss_static'] = geometry_weight * \
                    config.get('lambda_static', 0.2) * \
                    self._loss_static(output, cache)
                loss_dict[prefix + '.loss_dynamic'] = geometry_weight * \
                    config.get('lambda_dynamic', 0.5) * \
                    self._loss_dynamic(
                        cls_scores, output, cache, refine_pts_flat)
                loss_dict[prefix + '.loss_smooth'] = geometry_weight * \
                    config.get('lambda_smooth', 0.05) * \
                    self._loss_smooth(previous_output, output)
                loss_dict[prefix + '.loss_leak'] = geometry_weight * \
                    config.get('lambda_leak', 0.05) * \
                    self._loss_leak(output, cache)
                for name, value in self._role_diagnostics(
                        output, cache).items():
                    loss_dict[prefix + '.' + name] = value
                for name, value in self._residual_diagnostics(output).items():
                    loss_dict[prefix + '.' + name] = value
                for name, value in self._motion_diagnostics(output).items():
                    loss_dict[prefix + '.' + name] = value
                for name, value in self._geometry_diagnostics(
                        output, cache).items():
                    loss_dict[prefix + '.' + name] = value
                previous_output = output
        return loss_dict

    def loss_pretrain(self,voxel_semantic,temporal_semantics,temporal2ego, pred_dicts):
        B = voxel_semantic.shape[0]
        loss = dict()

        init_points_all = pred_dicts['init_points']

        pred_dict = {'init_points':init_points_all,
                     'all_cls_scores': list(),
                     'all_refine_pts': list(),
                     'all_pred_stamps': list()}
        num_refine = len(pred_dicts['all_cls_scores']) if self.pretrain else 1
        for cls_scores, refine_pts in zip(pred_dicts['all_cls_scores'][:num_refine], pred_dicts['all_refine_pts'][:num_refine]):
            pred_dict['all_cls_scores'].append(cls_scores)
            pred_dict['all_refine_pts'].append(refine_pts)

        voxel_semantic_stack,voxel_stamp_stack = self.get_sparse_voxels_stack(voxel_semantic,temporal_semantics,temporal2ego)


        loss.update(self.loss_stack(voxel_semantic_stack,pred_dict,time_stamp = 0,voxel_stamp=voxel_stamp_stack))
        
        return loss


    def loss_stack(self, voxel_semantics, preds_dicts, time_stamp=0,temporal2ego=None,voxel_stamp=None):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores']  # 6 ,B,2k4,32,17
        all_refine_pts = preds_dicts['all_refine_pts']

        num_dec_layers = len(all_cls_scores)

        gt_points_list, gt_labels_list, gt_stamps_list = \
            self.get_sparse_voxels_new(voxel_semantics,voxel_stamp)
        # gt_points_list,  gt_temporal_list = \
        #     self.get_sparse_voxels_new(voxel_stamp)
        if temporal2ego is not None:
            B = len(gt_points_list)
            gt_points_list = [torch.matmul(gt_points_list[b],temporal2ego[b,:3,:3].transpose(1,0)) + temporal2ego[b,None:3,3] for b in range(B)]

        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        # all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_stamps_list = [gt_stamps_list for _ in range(num_dec_layers)]

        if len(all_cls_scores)>0:
            losses_cls, losses_pts = multi_apply(
            self.loss_single_mask, all_cls_scores, all_refine_pts,
            all_gt_points_list, all_gt_labels_list,all_gt_stamps_list,temporal_reweight = True)
        else:
            losses_cls, losses_pts = [],[]
        loss_dict = dict()
        # loss of init_points
        if init_points is not None:
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)

            _, init_loss_pts = self.loss_single_mask(
                pseudo_scores, init_points, gt_points_list,
                 gt_labels_list,gt_stamps_list,temporal_reweight=True)
            loss_dict[f'init_loss_pts'] = init_loss_pts

        # loss from the last decoder layer

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls, losses_pts):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            num_dec_layer += 1
        return loss_dict
    def loss(self, voxel_semantics, preds_dicts):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores'] # 6 ,B,2k4,32,17
        all_refine_pts = preds_dicts['all_refine_pts']

        num_dec_layers = len(all_cls_scores)
        gt_points_list, gt_labels_list = \
            self.get_sparse_voxels(voxel_semantics)
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        # all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_pts = multi_apply(
            self.loss_single, all_cls_scores, all_refine_pts,
            all_gt_points_list,  all_gt_labels_list)

        loss_dict = dict()
        # loss of init_points
        if init_points is not None:
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)
            _, init_loss_pts = self.loss_single(
                pseudo_scores, init_points, gt_points_list,
                 gt_labels_list)
            loss_dict['init_loss_pts'] = init_loss_pts

        # loss from the last decoder layer


        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls, losses_pts):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            num_dec_layer += 1
        return loss_dict

    def get_occ(self, pred_dicts, expand_range = False,thre1 = 0.1,thre2=0.1):
        if expand_range:
            pc_range = self.pc_range_new
            voxel_num = self.voxel_num_new
        else:
            pc_range = self.pc_range
            voxel_num = self.voxel_num

        cls_scores = pred_dicts['cls_scores'].sigmoid()
        # cls_scores = all_cls_scores[-1].sigmoid()
        refine_pts = pred_dicts['refine_pts']
        if thre1!=None:
            mask = cls_scores.argmax(-1)==15
            dis = torch.norm(refine_pts - torch.mean(refine_pts,dim=2,keepdim=True),dim=-1)
            cls_scores[mask] = cls_scores[mask] * torch.clamp(thre1 /dis[mask],max=1)[:,None]
        if thre2!=None:
            mask = cls_scores.argmax(-1) == 16
            dis = torch.norm(refine_pts - torch.mean(refine_pts, dim=2, keepdim=True), dim=-1)
            cls_scores[mask] = cls_scores[mask] * torch.clamp(thre2 / dis[mask], max=1)[:, None]

        batch_size = refine_pts.shape[0]
        ctr_dist_thr = self.test_cfg.get('ctr_dist_thr', 3.0)
        score_thr = self.test_cfg.get('score_thr', 0.1)
        score_thr = torch.tensor(score_thr,device=cls_scores.device)
        result_list = []
        for i in range(batch_size):
            refine_pts, cls_scores = refine_pts[i], cls_scores[i]
            refine_pts = decode_points(refine_pts, self.pc_range)

            # filter weak points by distance and score
            centers = refine_pts.mean(dim=1, keepdim=True)
            ctr_dists = torch.norm(refine_pts - centers, dim=-1)
            mask_dist = ctr_dists < ctr_dist_thr
            max_score,index = cls_scores.max(-1)
            mask_score = (cls_scores.max(-1)[0] > score_thr[index])
            mask = mask_dist & mask_score
            refine_pts = refine_pts[mask]
            cls_scores = cls_scores[mask]
            if False:
                pts = torch.cat([refine_pts, cls_scores], dim=-1)
                pts_infos, voxels, num_pts = self.voxel_generator(pts)
                voxels = torch.flip(voxels, [1]).long()
                pts, scores = pts_infos[..., :3], pts_infos[..., 3:]
                scores = scores.sum(dim=1) / num_pts[..., None]
            else:
                index = ((refine_pts - pc_range[:3]) // self.voxel_size).long()
                mask = torch.logical_and(index >= 0, index < voxel_num).all(-1)
                voxels, unq_inv, pts_num = torch.unique(index[mask], return_inverse=True, return_counts=True, dim=0)
                scores = torch_scatter.scatter_max(cls_scores[mask], unq_inv, 0)[0]
            occ = scores.new_zeros((voxel_num[0], voxel_num[1],
                                    voxel_num[2], self.num_classes))
            occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = scores
            occ = occ.permute(3, 0, 1, 2).unsqueeze(0)
            if self.test_cfg.get('padding', True):
                # padding
                dilated_occ = F.max_pool3d(occ, 3, stride=1, padding=1)
                eroded_occ = -F.max_pool3d(-dilated_occ, 3, stride=1, padding=1)

                max_score,index = occ.max(1)
                original_mask = (max_score > score_thr[index]) #| (eroded_occ.argmax(1)==15 )
                original_mask = original_mask.expand_as(eroded_occ)
                eroded_occ[original_mask] = occ[original_mask]
            else:
                eroded_occ = occ
                # eroded_occ = occ[0].permute(1,2,3,0)
            eroded_occ = eroded_occ.squeeze(0).permute(1, 2, 3, 0)
            voxels = torch.nonzero((eroded_occ > score_thr).any(dim=-1))
            scores = eroded_occ[voxels[:, 0], voxels[:, 1], voxels[:, 2], :]
            occ_pred = torch.ones(voxel_num.tolist(), device=eroded_occ.device, dtype=torch.long) * 17
            occ_pred[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = eroded_occ[voxels[:, 0], voxels[:, 1],
                                                                     voxels[:, 2], :].argmax(-1)
            result_list.append(occ_pred)

        return torch.stack(result_list, 0)

    @torch.no_grad()
    def get_sparse_voxels_new(self, voxel_semantics,voxel_stamp):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size_new[0] + self.pc_range_new[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size_new[1] + self.pc_range_new[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size_new[2] + self.pc_range_new[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, H, Z)
        coors = torch.stack([xx, yy, zz], dim=-1)  # actual space

        gt_points, gt_masks, gt_labels, gt_stamps = [], [], [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.empty_label
            gt_points.append(coors[mask])
              # camera mask and not empty
            gt_labels.append(voxel_semantics[i][mask])
            gt_stamps.append(voxel_stamp[i][mask])

        return gt_points,  gt_labels, gt_stamps

    @torch.no_grad()
    def get_sparse_voxels(self, voxel_semantics):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, H, Z)
        coors = torch.stack([xx, yy, zz], dim=-1)  # actual space

        gt_points, gt_labels = [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.empty_label
            gt_points.append(coors[mask])
             # camera mask and not empty
            gt_labels.append(voxel_semantics[i][mask])

        return gt_points, gt_labels

    @torch.no_grad()
    def get_sparse_voxels_stack(self,voxel_semantic,temporal_semantics,temporal_pre2curs,mask_moving = True):
        B, W, H, Z = voxel_semantic.shape
        stack_semantics = voxel_semantic.new_ones(B,400,300,24) * 17
        stack_stamp = voxel_semantic.new_zeros(B,400,300,24,self.num_fu_frames+1)
        device = voxel_semantic.device
        new_voxel_num = self.voxel_num_new
        pc_range = self.pc_range_new
        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, W, Z)
        coors = torch.stack([xx, yy, zz], dim=-1)

        for b in range(B):
            for t in [0,6,5,4,3,2,1]:
                if t>0:
                    cur_semantic,fu2cur = (temporal_semantics[t]['voxel_semantics'][b],
                                       temporal_pre2curs[t-1][b])
                else:
                    cur_semantic,fu2cur = voxel_semantic[b],torch.eye(4,device=device)
                cur_points = coors[cur_semantic!=17]
                cur_semantic = cur_semantic[cur_semantic!=17]
                cur_points = torch.matmul(cur_points,fu2cur[:3,:3].transpose(0,1)) + fu2cur[:3,3]
                warped_coords = torch.floor((cur_points - pc_range[None,:3])//self.voxel_size[None,:]).long()
                mask = torch.logical_and(warped_coords>=0,warped_coords<new_voxel_num[None,:]).all(-1)

                stack_stamp[b, warped_coords[mask, 0], warped_coords[mask, 1], warped_coords[mask, 2], t] = 1
                # if mask_moving and t>0:
                #     mask = mask * torch.logical_or(cur_semantic<2,cur_semantic>10)

                stack_semantics[b,warped_coords[mask,0],warped_coords[mask,1],warped_coords[mask,2]] = cur_semantic[mask]

        return stack_semantics,stack_stamp
