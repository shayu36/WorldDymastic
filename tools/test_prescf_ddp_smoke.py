#!/usr/bin/env python3
"""Dataset-free two-GPU DDP smoke test for the six-step PreSCF recursion."""

import argparse
import os
import sys
from pathlib import Path

# ``torchrun`` may come from an editable project environment whose own source
# checkout precedes the launched script.  Always test this working tree.
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
from mmcv import Config
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from mmdet3d.models import build_model


class PreSCFSmokeWrapper(nn.Module):
    """Expose the detector's state transition through an ordinary DDP forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, query_feat, points, semantics, ego_feat):
        outputs = self.model._forward_prescf(
            dict(query_feat=query_feat, all_refine_pts=[points],
                 all_cls_scores=[semantics]),
            ego_feat, [{}], {})
        states = outputs['prescf_outputs']
        loss = query_feat.new_zeros(())
        for state in states:
            loss = loss + (
                state['points_metric'].square().mean()
                + state['semantics'].square().mean()
                + state['role_logits'].square().mean()
                + state['query_role'].square().mean()
                + state['query_motion'].square().mean()
                + state['predicted_pose'].square().mean())
        return loss / max(len(states), 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/sparseworld/nuscenes-temporal/sparseworld-traj-prescf.py')
    parser.add_argument('--iterations', type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device('cuda', local_rank)

    cfg = Config.fromfile(args.config)
    detector = build_model(
        cfg.model, train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).to(device).eval()
    detector.pretrain = False
    detector.pts_bbox_head.ind_stamps_all = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6], device=device)
    wrapper = DistributedDataParallel(
        PreSCFSmokeWrapper(detector), device_ids=[local_rank],
        output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in wrapper.parameters()
         if parameter.requires_grad), lr=1e-5)

    # Establish the rank-shared history snapshot before any rank-local
    # increments are applied.  Without this initial synchronization, the
    # first delta merge treats rank 0's checkpoint/history tensor as the
    # authoritative state and can silently drop rank 1's first update.
    detector._synchronize_tass_state(detector)
    initial_count_sum = detector.pts_bbox_head.num_stamps_all.detach().to(
        torch.int64).sum()

    batch, queries, num_points, channels = 1, 7, 48, 256
    for iteration in range(args.iterations):
        generator = torch.Generator(device=device)
        generator.manual_seed(iteration + 1000 * dist.get_rank())
        query_feat = torch.randn(
            batch, queries, channels, device=device, generator=generator)
        points = torch.rand(
            batch, queries, num_points, 3, device=device, generator=generator)
        semantics = torch.randn(
            batch, queries, num_points, 17, device=device,
            generator=generator)
        ego_feat = torch.randn(
            batch, 1, channels, device=device, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = wrapper(query_feat, points, semantics, ego_feat)
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite loss at iteration {}'.format(iteration))
        loss.backward()
        for name, parameter in wrapper.named_parameters():
            if parameter.requires_grad and parameter.grad is None:
                raise RuntimeError('unused trainable parameter: {}'.format(name))
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError('non-finite gradient: {}'.format(name))
        optimizer.step()
        # Emulate the rank-local target counts updated by OPUSHead losses.
        # Synchronizing deltas must preserve all contributions without
        # multiplying the previously shared history.
        tass_counts = detector.pts_bbox_head.num_stamps_all
        tass_counts[iteration % tass_counts.shape[0], dist.get_rank()] += 1
        detector._synchronize_tass_state(detector)
        reference_counts = tass_counts.clone()
        dist.broadcast(reference_counts, src=0)
        if not torch.equal(tass_counts, reference_counts):
            raise RuntimeError('TASS count synchronization failed')

    dist.barrier()
    final_count_sum = detector.pts_bbox_head.num_stamps_all.detach().to(
        torch.int64).sum()
    expected_count_sum = initial_count_sum + args.iterations * dist.get_world_size()
    if final_count_sum.item() != expected_count_sum.item():
        raise RuntimeError(
            'TASS count delta mismatch: got {}, expected {}'.format(
                final_count_sum.item(), expected_count_sum.item()))
    if dist.get_rank() == 0:
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        print('PreSCF DDP smoke passed: {} iterations, {} ranks, '
              'peak {:.1f} MiB/rank'.format(
                  args.iterations, dist.get_world_size(), peak))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
