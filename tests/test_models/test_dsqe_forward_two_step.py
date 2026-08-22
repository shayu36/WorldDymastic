import numpy as np
import pytest
import torch
from mmcv import Config

from mmdet3d.models import build_model


def _sparse_semantics(batch_size, step):
    semantics = torch.full(
        (batch_size, 200, 200, 16), 17, dtype=torch.long)
    start = 80 + step
    semantics[:, start:start + 4, 96:100, 7:9] = 4
    semantics[:, 105:109, 110:114, 7:9] = 11
    return semantics


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='full DSQE smoke test requires CUDA')
def test_two_step_dsqe_forward_with_role_correction_and_dynamic_loss():
    cfg = Config.fromfile(
        'configs/sparseworld/nuscenes-temporal/'
        'sparseworld-traj-finetune.py')
    model = build_model(
        cfg.model, train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    legacy_cls_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith('cls_branch.')
    ]
    assert len(legacy_cls_parameters) == 6
    assert not any(
        parameter.requires_grad for parameter in legacy_cls_parameters)
    model.init_weights()
    model.set_epoch(cfg.finetune_epoch + 1)
    model.cuda().train()

    batch_size = 1
    num_images = cfg.num_frames * 6
    height, width = 64, 96
    img = torch.randn(
        batch_size, num_images, 3, height, width, device='cuda')

    identity = np.eye(4, dtype=np.float32)
    img_metas = [{
        'ego2lidar': identity,
        'ego2global': identity,
        'lidar2img': np.tile(identity[None], (num_images, 1, 1)),
    }]

    temporal_ego_states = {
        index: torch.randn(batch_size, 1, 21, device='cuda')
        for index in range(6)}
    temporal_trajs = torch.randn(batch_size, 6, 2, device='cuda')
    temporal_ego2global = {
        index: np.tile(identity[None], (batch_size, 1, 1))
        for index in range(6)}
    temporal2ego = {
        index: torch.eye(4, device='cuda').unsqueeze(0)
        for index in range(6)}
    temporal_semantics = {
        index: {
            'voxel_semantics': _sparse_semantics(
                batch_size, index).cuda(),
            'mask_lidar': torch.ones(
                batch_size, 200, 200, 16, dtype=torch.bool,
                device='cuda'),
            'mask_camera': torch.ones(
                batch_size, 200, 200, 16, dtype=torch.bool,
                device='cuda'),
        }
        for index in range(1, 7)
    }

    voxel_semantics = _sparse_semantics(batch_size, 0).cuda()
    mask_lidar = torch.ones_like(voxel_semantics, dtype=torch.bool)
    mask_camera = torch.ones_like(voxel_semantics, dtype=torch.bool)

    losses = model(
        return_loss=True,
        img=img,
        img_metas=img_metas,
        voxel_semantics=voxel_semantics,
        mask_lidar=mask_lidar,
        mask_camera=mask_camera,
        temporal_ego_states=temporal_ego_states,
        temporal_trajs=temporal_trajs,
        temporal_ego2global=temporal_ego2global,
        temporal2ego=temporal2ego,
        temporal_semantics=temporal_semantics)

    assert 'loss_total' not in losses
    for key in ('fu1.loss_role', 'fu1.loss_dynamic',
                'fu2.loss_role', 'fu2.loss_dynamic'):
        assert key in losses
        assert torch.isfinite(losses[key]).all()

    loss_total = sum(
        value.mean() for key, value in losses.items() if 'loss' in key)
    assert torch.isfinite(loss_total)
    loss_total.backward()

    role_gradients = [
        parameter.grad for parameter in model.role_router.parameters()
        if parameter.grad is not None
    ]
    assert role_gradients
    assert all(torch.isfinite(gradient).all()
               for gradient in role_gradients)

    missing_gradients = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing_gradients

    for prefix in (
            'role_router.', 'yaw_head.', 'dual_evolution.',
            'dual_interaction.', 'joint_refine.',
            'semantic_correction_head.'):
        gradients = [
            parameter.grad for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.requires_grad
        ]
        assert gradients, prefix
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


if __name__ == '__main__':
    test_two_step_dsqe_forward_with_role_correction_and_dynamic_loss()
