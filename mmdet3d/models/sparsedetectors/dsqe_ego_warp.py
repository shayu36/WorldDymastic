"""Rigid ego-frame transforms used by DSQE-PreSCF."""

import torch
import torch.nn.functional as F
from torch import nn

from .bbox.utils import decode_points, encode_points


class DSQEEgoWarp(nn.Module):
    def __init__(self, pc_range, frame_mode='future_ego', eps=1e-6):
        super().__init__()
        if frame_mode not in ('future_ego', 't0_aligned'):
            raise ValueError('Unsupported frame_mode: {}'.format(frame_mode))
        self.frame_mode = frame_mode
        self.eps = eps
        self.register_buffer('pc_range', torch.as_tensor(pc_range).float())

    @staticmethod
    def identity(batch_size, device, dtype):
        return torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)

    def pose_to_matrix(self, pose):
        xy = pose[..., :2]
        sin_cos = F.normalize(pose[..., 2:4], dim=-1, eps=self.eps)
        sin_yaw, cos_yaw = sin_cos.unbind(-1)
        matrix = torch.zeros(*pose.shape[:-1], 4, 4, device=pose.device, dtype=pose.dtype)
        matrix[..., 0, 0] = cos_yaw
        matrix[..., 0, 1] = -sin_yaw
        matrix[..., 1, 0] = sin_yaw
        matrix[..., 1, 1] = cos_yaw
        matrix[..., 2, 2] = 1
        matrix[..., 3, 3] = 1
        matrix[..., :2, 3] = xy
        return matrix

    @staticmethod
    def inverse(matrix):
        rot = matrix[..., :3, :3]
        trans = matrix[..., :3, 3]
        out = torch.zeros_like(matrix)
        out[..., :3, :3] = rot.transpose(-1, -2)
        out[..., :3, 3] = -torch.matmul(rot.transpose(-1, -2), trans.unsqueeze(-1)).squeeze(-1)
        out[..., 3, 3] = 1
        return out

    @staticmethod
    def matrix_to_pose(matrix):
        yaw = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
        return torch.stack([matrix[..., 0, 3], matrix[..., 1, 3],
                            yaw.sin(), yaw.cos()], dim=-1)

    @staticmethod
    def compose(first, second):
        return torch.matmul(first, second)

    @staticmethod
    def t0_displacement_to_current(displacement, current_to_t0):
        rotation = current_to_t0[..., :2, :2]
        return torch.matmul(rotation.transpose(-1, -2),
                            displacement.unsqueeze(-1)).squeeze(-1)

    def trajectory_to_ego_relative(self, displacement_t0_lidar,
                                   current_ego_to_t0, lidar_to_ego,
                                   yaw_sin_cos):
        ego_to_lidar = self.inverse(lidar_to_ego)
        current_lidar_to_t0 = self.compose(
            self.compose(ego_to_lidar, current_ego_to_t0), lidar_to_ego)
        current_delta = self.t0_displacement_to_current(
            displacement_t0_lidar, current_lidar_to_t0)
        delta3 = torch.cat([current_delta,
                            current_delta.new_zeros(*current_delta.shape[:-1], 1)], -1)
        yaw_sin_cos = F.normalize(yaw_sin_cos, dim=-1, eps=self.eps)
        zero = yaw_sin_cos.new_zeros(*yaw_sin_cos.shape[:-1], 2)
        ego_rotation = self.pose_to_matrix(torch.cat([zero, yaw_sin_cos], -1))[..., :3, :3]
        ext_rot = lidar_to_ego[..., :3, :3]
        ext_trans = lidar_to_ego[..., :3, 3]
        lidar_delta = torch.matmul(ext_rot, delta3.unsqueeze(-1)).squeeze(-1)
        rotated_arm = torch.matmul(ego_rotation, ext_trans.unsqueeze(-1)).squeeze(-1)
        ego_translation = lidar_delta + ext_trans - rotated_arm
        relative_pose = torch.cat([ego_translation[..., :2], yaw_sin_cos], -1)
        return current_delta, relative_pose, self.pose_to_matrix(relative_pose)

    def build_relative_targets(self, ego2global_sequence):
        current = ego2global_sequence[:, :-1]
        nxt = ego2global_sequence[:, 1:]
        next_to_current = torch.matmul(self.inverse(current), nxt)
        pose = self.matrix_to_pose(next_to_current)
        return self.pose_to_matrix(pose), pose

    @staticmethod
    def transform_metric(points, transform):
        rot = transform[..., :3, :3]
        trans = transform[..., :3, 3]
        # Homogeneous transforms follow the usual column convention while
        # points are stored with a trailing coordinate dimension.  The
        # ``bij`` einsum layout below is equivalent to ``points @ R.T``.
        out = torch.einsum('b...j,bij->b...i', points, rot)
        view = [trans.shape[0]] + [1] * (points.ndim - 2) + [3]
        return out + trans.reshape(view)

    def warp_encoded(self, points, transform):
        return encode_points(self.transform_metric(decode_points(points, self.pc_range), transform), self.pc_range)

    def current_to_next(self, points, next_to_current):
        return points if self.frame_mode == 't0_aligned' else self.warp_encoded(points, self.inverse(next_to_current))

    def t0_to_next(self, points, next_to_t0):
        return points if self.frame_mode == 't0_aligned' else self.warp_encoded(points, self.inverse(next_to_t0))
