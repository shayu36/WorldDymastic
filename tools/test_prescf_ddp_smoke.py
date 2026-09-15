#!/usr/bin/env python3
"""Real ``forward_train`` DDP smoke test for DSQE-PreSCF.

This is intentionally a short synthetic-data integration test, not a
surrogate state-transition loss. It constructs the same tensor/metadata
contracts as ``NuscenesDatasetOccTrajectory`` and calls the detector's actual
``forward_train`` so occupancy, actor-role, ego-pose, geometry and planning
losses all participate in the backward graph. It still needs two CUDA
devices and is therefore normally run by the user locally.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
from mmcv import Config
from torch.nn.parallel import DistributedDataParallel

from mmdet3d.models import build_model


def _identity_batch(batch, device, translation=0.0):
    matrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch, 1, 1)
    matrix[:, 0, 3] = translation
    return matrix


def make_batch(batch, device, image_height, image_width, num_frames=5,
               future_steps=6):
    """Build a batch with two different ragged actor counts."""
    img = torch.randn(batch, num_frames * 6, 3, image_height, image_width,
                      device=device)
    voxel = torch.full((batch, 200, 200, 16), 17, dtype=torch.long,
                       device=device)
    voxel[:, 100, 100, 2] = 4
    mask_camera = torch.ones_like(voxel, dtype=torch.bool)
    temporal_semantics = {}
    temporal2ego = {}
    temporal_adjacent2ego = {}
    temporal_ego2global = {}
    temporal_ego_states = {}
    for step in range(future_steps):
        future = voxel.clone()
        future[:, 100 + step, 100, 2] = 4
        temporal_semantics[step + 1] = {'voxel_semantics': future}
        cumulative = _identity_batch(batch, device, 0.1 * (step + 1))
        previous = _identity_batch(batch, device, 0.1 * step)
        temporal2ego[step] = cumulative
        temporal_adjacent2ego[step] = torch.linalg.inv(previous) @ cumulative
        temporal_ego2global[step] = torch.linalg.inv(cumulative)
        temporal_ego_states[step] = torch.zeros(batch, 21, device=device)

    actor_counts = (2, 3)
    boxes, feats, labels, lidar2ego = [], [], [], []
    for sample, count in enumerate(actor_counts[:batch]):
        sample_boxes = torch.zeros(count, 9, device=device)
        sample_boxes[:, 0] = torch.arange(count, device=device) + sample
        sample_boxes[:, 3:6] = torch.tensor(
            [2., 4., 2.], device=device).expand(count, 3)
        sample_feats = torch.zeros(count, 34, device=device)
        sample_feats[:, 2] = 1.0
        sample_feats[:, 12:18] = 1.0
        sample_labels = torch.full((count,), 4, dtype=torch.long,
                                   device=device)
        boxes.append(sample_boxes)
        feats.append(sample_feats)
        labels.append(sample_labels)
        lidar2ego.append(torch.eye(4, device=device))

    # OPUSTransformer consumes the same camera calibration metadata as the
    # real Collect4D pipeline.  Keep one identity projection per camera and
    # include shape/padding fields that the image encoder updates in-place.
    # Keep image metadata on CPU as in a real DataLoader.  OPUSTransformer
    # converts ``lidar2img`` through NumPy while model tensors remain on CUDA.
    lidar2img = torch.eye(4).unsqueeze(0).repeat(num_frames * 6, 1, 1)
    img_metas = [
        {
            'ego2lidar': torch.eye(4).numpy(),
            'ego2global': torch.eye(4).numpy(),
            'lidar2img': lidar2img.clone().numpy(),
            'filename': ['synthetic'] * (num_frames * 6),
            'img_shape': [(image_height, image_width, 3)] * (num_frames * 6),
            'ori_shape': [(image_height, image_width, 3)] * (num_frames * 6),
            'pad_shape': [(image_height, image_width, 3)] * (num_frames * 6),
        }
        for _ in range(batch)]
    return dict(
        img=img, img_metas=img_metas, voxel_semantics=voxel,
        mask_camera=mask_camera, temporal_semantics=temporal_semantics,
        temporal2ego=temporal2ego,
        temporal_adjacent2ego=temporal_adjacent2ego,
        temporal_ego2global=temporal_ego2global,
        temporal_ego_states=temporal_ego_states,
        temporal_trajs=torch.zeros(batch, future_steps, 2, device=device),
        temporal_agent_boxes=boxes, temporal_agent_feats=feats,
        temporal_agent_labels=labels, temporal_agent_lidar2ego=lidar2ego)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=
        'configs/sparseworld/nuscenes-temporal/sparseworld-traj-prescf.py')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--stage', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--image-height', type=int, default=64)
    parser.add_argument('--image-width', type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device('cuda', local_rank)
    cfg = Config.fromfile(args.config)
    detector = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg')).to(device)
    stage_epoch = (0 if args.stage == 1 else
                   detector.prescf_stage1_end_epoch if args.stage == 2 else
                   detector.dsqe_cfg.get('stage3_start_epoch', 16))
    detector.set_epoch(stage_epoch)
    detector.train()
    if args.stage in (1, 2):
        assert all(not p.requires_grad for p in detector.img_backbone.parameters())
        assert all(not p.requires_grad for p in detector.img_neck.parameters())
    wrapper = DistributedDataParallel(
        detector, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        [p for p in wrapper.parameters() if p.requires_grad], lr=1e-5)
    batch = make_batch(2, device, args.image_height, args.image_width)

    for iteration in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        losses = wrapper(**batch)
        tensor_losses = [value for value in losses.values()
                         if torch.is_tensor(value) and value.requires_grad]
        if not tensor_losses:
            raise RuntimeError('forward_train returned no differentiable loss')
        loss = sum(tensor_losses)
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite loss at iteration {}'.format(iteration))
        loss.backward()
        for name, parameter in wrapper.named_parameters():
            if not parameter.requires_grad:
                continue
            # Stage 1/2 keeps final TASS layers in the reducer with a zero
            # gradient hook; they are intentionally excluded here.
            if name.startswith('module.pts_bbox_head.') and args.stage < 3:
                continue
            if parameter.grad is None:
                raise RuntimeError('unused expected parameter: {}'.format(name))
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError('non-finite gradient: {}'.format(name))
        optimizer.step()

    dist.barrier()
    if dist.get_rank() == 0:
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        print('PreSCF real forward_train DDP smoke passed: stage={}, '
              '{} iterations, {} ranks, peak {:.1f} MiB/rank'.format(
                  args.stage, args.iterations, dist.get_world_size(), peak))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
