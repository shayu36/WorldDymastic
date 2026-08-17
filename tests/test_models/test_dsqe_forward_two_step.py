import torch
import numpy as np
from mmcv import Config
from mmdet3d.models import build_model


def test_two_step_dsqe_forward_with_role_correction_and_dynamic_loss():
    cfg = Config.fromfile(
        'configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py')
    model = build_model(
        cfg.model, train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.set_epoch(cfg.finetune_epoch)
    model.train()

    batch_size = 2
    num_cams = 6
    h, w = 256, 704
    img = torch.randn(batch_size, num_cams, 3, h, w)

    img_metas = [{
        'ego2lidar': np.eye(4, dtype=np.float32),
        'ego2global': np.eye(4, dtype=np.float32),
        'lidar2img': np.tile(np.eye(4, dtype=np.float32)[None, :, :], (6, 1, 1)),
    } for _ in range(batch_size)]

    temporal_ego_states = {
        i: torch.randn(batch_size, 1, 21) for i in range(6)}
    temporal_trajs = torch.randn(batch_size, 6, 2)
    temporal_ego2global = {
        i: np.tile(np.eye(4, dtype=np.float32)[None, :, :], (batch_size, 1, 1))
        for i in range(6)
    }
    temporal2ego = {
        i: np.tile(np.eye(4, dtype=np.float32)[None, :, :], (batch_size, 1, 1))
        for i in range(6)
    }
    temporal_semantics = {
        i: {
            'voxel_semantics': torch.randint(
                0, 18, (batch_size, 200, 200, 16)),
            'mask_lidar': torch.ones(batch_size, 200, 200, 16, dtype=torch.bool),
            'mask_camera': torch.ones(
                batch_size, 200, 200, 16, dtype=torch.bool),
        }
        for i in range(1, 3)
    }

    voxel_semantics = torch.randint(0, 18, (batch_size, 200, 200, 16))
    mask_lidar = torch.ones(batch_size, 200, 200, 16, dtype=torch.bool)
    mask_camera = torch.ones(batch_size, 200, 200, 16, dtype=torch.bool)

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
        temporal_semantics=temporal_semantics
    )

    assert 'loss_total' in losses
    assert 'fu1.loss_role' in losses
    assert 'fu1.loss_dynamic' in losses
    assert 'fu2.loss_role' in losses
    assert 'fu2.loss_dynamic' in losses
    assert torch.isfinite(losses['loss_total'])
    print('loss_total:', losses['loss_total'].item())
    print('loss keys:', len([k for k in losses if k.startswith('fu')]))

    losses['loss_total'].backward()
    role_grad_finite = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.role_router.parameters()
    )
    print('role_router grad finite:', role_grad_finite)
    assert role_grad_finite


if __name__ == '__main__':
    test_two_step_dsqe_forward_with_role_correction_and_dynamic_loss()
    print('Test passed')
