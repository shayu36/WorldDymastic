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
import json
import os
import sys
from collections import OrderedDict
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


EXPECTED_PRESCF_MODULES = OrderedDict([
    ('role_router', ('role_router.',)),
    ('ego_pose_head', ('prescf_ego_pose_head.',)),
    ('motion_head', ('dual_evolution.motion_head.',)),
    ('static_evolution', ('dual_evolution.static_head.',)),
    ('dynamic_evolution', ('dual_evolution.dynamic_head.',)),
    ('new_query_evolution', ('dual_evolution.new_head.',)),
    ('dual_interaction', ('dual_interaction.',)),
    ('joint_refine', (
        'joint_refine.dynamic_proj.', 'joint_refine.static_proj.',
        'joint_refine.point_proj.', 'joint_refine.norm.',
        'joint_refine.ffn.', 'joint_refine.spatial_attn.',
        'joint_refine.spatial_norm.')),
    ('point_correction', ('joint_refine.point_correction.',)),
    ('role_correction', ('joint_refine.role_correction.',)),
    ('absolute_semantic_head', ('joint_refine.semantic_head.',)),
    ('point_semantic_adapter', (
        'joint_refine.point_adapter.', 'joint_refine.point_adapter_gate')),
    ('state_embeddings', ('source_embedding.', 'activation_embedding.')),
    ('dual_stream_planning', (
        'ego_cross_attn_dynamic.', 'ego_cross_attn_static.',
        'ego_dynamic_proj.', 'ego_static_proj.', 'ego_fusion_norm.')),
])


def _staged_tass_parameter_names(detector):
    layers = list(
        detector.pts_bbox_head.transformer.decoder.decoder_layers)
    count = int(detector.dsqe_cfg.get('unfreeze_tass_layers', 2))
    staged_ids = {
        id(parameter)
        for layer in layers[-count:] if count > 0
        for parameter in layer.parameters()
    }
    return {
        name for name, parameter in detector.named_parameters()
        if id(parameter) in staged_ids
    }


def _group_parameter_names(detector):
    named = dict(detector.named_parameters())
    groups = OrderedDict()
    for group, prefixes in EXPECTED_PRESCF_MODULES.items():
        groups[group] = [
            name for name in named
            if any(name.startswith(prefix) for prefix in prefixes)
            and named[name].requires_grad
        ]
        if not groups[group]:
            raise RuntimeError(
                'expected trainable module has no parameters: {}'.format(group))
    staged = _staged_tass_parameter_names(detector)
    allowed = set(staged)
    for names in groups.values():
        allowed.update(names)
    unexpected = [
        name for name, parameter in named.items()
        if parameter.requires_grad and name not in allowed
    ]
    if unexpected:
        raise RuntimeError(
            'trainable parameters outside smoke-test allowlist: {}'.format(
                unexpected))
    return groups, staged


def build_smoke_optimizer(detector, learning_rate=1e-5):
    """Build explicit main/Stage-3 TASS groups for the real smoke test."""
    staged_names = _staged_tass_parameter_names(detector)
    main, staged = [], []
    for name, parameter in detector.named_parameters():
        if not parameter.requires_grad:
            continue
        (staged if name in staged_names else main).append(parameter)
    groups = [dict(
        params=main, lr=learning_rate, weight_decay=1e-2,
        group_name='prescf')]
    if staged:
        groups.append(dict(
            params=staged, lr=learning_rate * 0.1, weight_decay=0.0,
            group_name='tass_stage3'))
    return torch.optim.AdamW(groups), staged_names


def _gradient_stats(detector, names):
    named = dict(detector.named_parameters())
    parameters = [named[name] for name in names]
    gradients = [parameter.grad for parameter in parameters]
    total = sum(parameter.numel() for parameter in parameters)
    nonzero = sum(int((gradient != 0).sum()) for gradient in gradients
                  if gradient is not None)
    finite = all(gradient is not None and torch.isfinite(gradient).all()
                 for gradient in gradients)
    return dict(
        grad_is_none=any(gradient is None for gradient in gradients),
        grad_is_finite=bool(finite),
        grad_abs_sum=sum(float(gradient.abs().sum()) for gradient in gradients
                         if gradient is not None),
        nonzero_ratio=nonzero / max(total, 1))


def _changed_since(detector, names, before_step):
    named = dict(detector.named_parameters())
    return any(not torch.equal(named[name].detach(), before_step[name])
               for name in names)


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
    # A sparse static lattice keeps road supervision available throughout the
    # scene, while the compact actor patch guarantees non-empty dynamic role
    # and geometry targets on both ragged samples.
    voxel[:, ::10, ::10, 2] = 11
    voxel[:, 96:105, 96:105, 2:4] = 4
    mask_camera = torch.ones_like(voxel, dtype=torch.bool)
    temporal_semantics = {}
    temporal2ego = {}
    temporal_adjacent2ego = {}
    temporal_ego2global = {}
    temporal_ego_states = {}
    for step in range(future_steps):
        future = voxel.clone()
        future[:, 96 + step:105 + step, 96:105, 2:4] = 4
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
            [6., 6., 3.], device=device).expand(count, 3)
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
    module_groups, staged_tass_names = _group_parameter_names(detector)
    all_tass_names = {
        name for name, _ in detector.named_parameters()
        if name.startswith('pts_bbox_head.')
    }
    nonstaged_tass = all_tass_names - staged_tass_names
    named_parameters = dict(detector.named_parameters())
    if any(named_parameters[name].requires_grad for name in nonstaged_tass):
        raise RuntimeError('non-configured TASS parameters are trainable')
    if any(not named_parameters[name].requires_grad
           for name in staged_tass_names):
        raise RuntimeError('configured Stage-3 TASS parameters are absent '
                           'from the DDP reducer')
    wrapper = DistributedDataParallel(
        detector, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=False)
    # DDP broadcasts rank-0 parameters during construction.  Snapshot the
    # frozen anchor only after that one-time synchronization; otherwise the
    # normal rank-0 -> rank-N broadcast is misreported as an optimizer update.
    frozen_tass_snapshot = {
        name: named_parameters[name].detach().cpu().clone()
        for name in sorted(nonstaged_tass)
    }
    optimizer, optimizer_staged_names = build_smoke_optimizer(detector)
    if optimizer_staged_names != staged_tass_names:
        raise RuntimeError('optimizer Stage-3 TASS group mismatch')
    tass_group = next(
        group for group in optimizer.param_groups
        if group.get('group_name') == 'tass_stage3')
    if (abs(tass_group['lr'] - 1e-6) > 1e-12 or
            tass_group['weight_decay'] != 0.0):
        raise RuntimeError('Stage-3 TASS optimizer group must use 0.1x LR '
                           'and zero weight decay')
    batch = make_batch(2, device, args.image_height, args.image_width)
    ever_nonzero = {group: False for group in module_groups}
    ever_changed = {group: False for group in module_groups}
    last_stats = {}
    staged_ever_nonzero = False
    staged_ever_changed = False
    required_losses = (
        '.loss_cls', '.loss_pts', '.loss_role', '.loss_ego',
        '.loss_static', '.loss_dynamic', '.loss_smooth', '.loss_leak',
        'loss_traj_')

    for iteration in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        losses = wrapper(**batch)
        for required in required_losses:
            if not any(required in name for name in losses):
                raise RuntimeError(
                    'forward_train omitted required real loss: {}'.format(
                        required))
        if args.stage == 3 and not any(
                name.startswith('fu6.') for name in losses):
            raise RuntimeError('Stage 3 did not execute the six-step rollout')
        tensor_losses = [value for value in losses.values()
                         if torch.is_tensor(value) and value.requires_grad]
        if not tensor_losses:
            raise RuntimeError('forward_train returned no differentiable loss')
        loss = sum(tensor_losses)
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite loss at iteration {}'.format(iteration))
        loss.backward()
        for name, parameter in detector.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                raise RuntimeError('unused expected parameter: {}'.format(name))
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError('non-finite gradient: {}'.format(name))
        for group, names in module_groups.items():
            stats = _gradient_stats(detector, names)
            last_stats[group] = stats
            if stats['grad_is_none'] or not stats['grad_is_finite']:
                raise RuntimeError(
                    'invalid gradients for {}: {}'.format(group, stats))
            ever_nonzero[group] |= stats['grad_abs_sum'] > 0 and \
                stats['nonzero_ratio'] > 0
        staged_stats = _gradient_stats(detector, sorted(staged_tass_names))
        staged_ever_nonzero |= staged_stats['grad_abs_sum'] > 0 and \
            staged_stats['nonzero_ratio'] > 0
        before_step = {
            name: parameter.detach().clone()
            for name, parameter in detector.named_parameters()
            if parameter.requires_grad
        }
        optimizer.step()
        for group, names in module_groups.items():
            changed = _changed_since(detector, names, before_step)
            ever_changed[group] |= (
                changed and last_stats[group]['grad_abs_sum'] > 0)
            last_stats[group]['parameter_changed_after_step'] = changed
        staged_changed = _changed_since(
            detector, sorted(staged_tass_names), before_step)
        staged_ever_changed |= staged_changed
        if args.stage < 3:
            if staged_stats['grad_abs_sum'] != 0 or staged_changed:
                raise RuntimeError(
                    'Stage {0} changed frozen staged TASS parameters'.format(
                        args.stage))

    # One iteration is useful as a quick wiring check.  The full acceptance
    # criterion starts at iteration 2 so zero-initialized adapters get one
    # optimizer step before their preceding layers are required to activate.
    if args.iterations >= 2:
        missing_gradient = [
            group for group, seen in ever_nonzero.items() if not seen]
        missing_change = [
            group for group, changed in ever_changed.items() if not changed]
        if missing_gradient:
            raise RuntimeError(
                'modules never received effective gradients: {}'.format(
                    missing_gradient))
        if missing_change:
            raise RuntimeError(
                'modules never changed after optimizer step: {}'.format(
                    missing_change))
        if args.stage == 3 and (
                not staged_ever_nonzero or not staged_ever_changed):
            raise RuntimeError(
                'Stage-3 TASS layers lacked nonzero gradients or updates')
    for name, reference in frozen_tass_snapshot.items():
        if not torch.equal(named_parameters[name].detach().cpu(), reference):
            raise RuntimeError(
                'non-configured TASS parameter changed: {}'.format(name))

    dist.barrier()
    if dist.get_rank() == 0:
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        report = {
            group: dict(
                last_stats[group],
                ever_nonzero_gradient=ever_nonzero[group],
                parameter_changed_at_least_once=ever_changed[group])
            for group in module_groups
        }
        report['tass_stage3_group'] = dict(
            grad_is_none=staged_stats['grad_is_none'],
            grad_is_finite=staged_stats['grad_is_finite'],
            grad_abs_sum=staged_stats['grad_abs_sum'],
            nonzero_ratio=staged_stats['nonzero_ratio'],
            parameter_changed_after_step=staged_changed,
            ever_nonzero_gradient=staged_ever_nonzero,
            parameter_changed_at_least_once=staged_ever_changed,
            expected_frozen=args.stage < 3,
            learning_rate=tass_group['lr'])
        report['nonstaged_tass'] = dict(
            requires_grad=False, parameter_changed_after_step=False,
            parameter_count=len(frozen_tass_snapshot))
        print(json.dumps(report, indent=2, sort_keys=True))
        print('PreSCF real forward_train DDP smoke passed: stage={}, '
              '{} iterations, {} ranks, peak {:.1f} MiB/rank'.format(
                  args.stage, args.iterations, dist.get_world_size(), peak))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
