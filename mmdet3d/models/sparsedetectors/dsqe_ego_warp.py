import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.sparsedetectors.bbox.utils import (
    decode_points, encode_points)


class DSQEEgoWarp(nn.Module):
    """Rigid ego-frame transforms used by DSQE-SCF.

    Pose matrices follow ``T_next_to_current``.  They map points expressed in
    the next ego frame into the current ego frame.  Scene propagation therefore
    applies their inverse to map current-frame points into the next frame.
    """

    def __init__(self, pc_range, frame_mode='future_ego', eps=1e-6):
        super().__init__()
        if frame_mode not in ('future_ego', 't0_aligned'):
            raise ValueError('Unsupported frame_mode: {}'.format(frame_mode))
        self.frame_mode = frame_mode
        self.eps = eps
        self.register_buffer('pc_range', torch.as_tensor(pc_range).float())

    @staticmethod
    def identity(batch_size, device, dtype):
        return torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(
            batch_size, 1, 1)

    def pose_to_matrix(self, pose):
        """Convert ``[dx, dy, sin(yaw), cos(yaw)]`` to a rigid matrix."""
        translation = pose[..., :2]
        sin_cos = F.normalize(pose[..., 2:4], dim=-1, eps=self.eps)
        sin_yaw, cos_yaw = sin_cos.unbind(dim=-1)

        matrix = torch.zeros(
            *pose.shape[:-1], 4, 4, device=pose.device, dtype=pose.dtype)
        matrix[..., 0, 0] = cos_yaw
        matrix[..., 0, 1] = -sin_yaw
        matrix[..., 1, 0] = sin_yaw
        matrix[..., 1, 1] = cos_yaw
        matrix[..., 2, 2] = 1
        matrix[..., 3, 3] = 1
        matrix[..., 0, 3] = translation[..., 0]
        matrix[..., 1, 3] = translation[..., 1]
        return matrix

    @staticmethod
    def matrix_to_pose(matrix):
        yaw = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
        return torch.stack([
            matrix[..., 0, 3], matrix[..., 1, 3], yaw.sin(), yaw.cos()
        ], dim=-1)

    @staticmethod
    def inverse(matrix):
        rotation = matrix[..., :3, :3]
        translation = matrix[..., :3, 3]
        rotation_inv = rotation.transpose(-1, -2)
        translation_inv = -torch.matmul(
            rotation_inv, translation.unsqueeze(-1)).squeeze(-1)

        output = torch.zeros_like(matrix)
        output[..., :3, :3] = rotation_inv
        output[..., :3, 3] = translation_inv
        output[..., 3, 3] = 1
        return output

    @staticmethod
    def compose(first, second):
        """Compose transforms as ``first @ second``."""
        return torch.matmul(first, second)

    @staticmethod
    def t0_displacement_to_current(displacement, current_to_t0):
        """Rotate fixed-t0 planar displacement into the current frame."""
        rotation = current_to_t0[..., :2, :2]
        return torch.matmul(
            rotation.transpose(-1, -2),
            displacement.unsqueeze(-1)).squeeze(-1)

    def trajectory_to_ego_relative(self,
                                   displacement_t0_lidar,
                                   current_ego_to_t0,
                                   lidar_to_ego,
                                   yaw_sin_cos):
        """Convert a fixed-L0 trajectory delta to an ego relative pose.

        ``displacement_t0_lidar`` is the displacement between consecutive
        LiDAR origins expressed in the fixed t0 LiDAR axes.  Scene queries are
        expressed in ego axes, so the displacement is first rotated into the
        current LiDAR frame and then converted with the per-sample LiDAR-to-ego
        extrinsic.  The latter includes the sensor lever-arm correction for the
        predicted ego yaw.
        """
        ego_to_lidar = self.inverse(lidar_to_ego)
        current_lidar_to_t0 = self.compose(
            self.compose(ego_to_lidar, current_ego_to_t0), lidar_to_ego)
        displacement_current_lidar = self.t0_displacement_to_current(
            displacement_t0_lidar, current_lidar_to_t0)
        displacement_current_lidar_3d = torch.cat([
            displacement_current_lidar,
            displacement_current_lidar.new_zeros(
                *displacement_current_lidar.shape[:-1], 1)
        ], dim=-1)

        yaw_sin_cos = F.normalize(yaw_sin_cos, dim=-1, eps=self.eps)
        zero_translation = yaw_sin_cos.new_zeros(
            *yaw_sin_cos.shape[:-1], 2)
        rotation_pose = torch.cat(
            [zero_translation, yaw_sin_cos], dim=-1)
        ego_rotation = self.pose_to_matrix(rotation_pose)[..., :3, :3]

        extrinsic_rotation = lidar_to_ego[..., :3, :3]
        extrinsic_translation = lidar_to_ego[..., :3, 3]
        lidar_delta_in_ego = torch.matmul(
            extrinsic_rotation,
            displacement_current_lidar_3d.unsqueeze(-1)).squeeze(-1)
        rotated_lever_arm = torch.matmul(
            ego_rotation,
            extrinsic_translation.unsqueeze(-1)).squeeze(-1)
        ego_translation = lidar_delta_in_ego + \
            extrinsic_translation - rotated_lever_arm

        relative_pose = torch.cat([
            ego_translation[..., :2], yaw_sin_cos
        ], dim=-1)
        relative_matrix = self.pose_to_matrix(relative_pose)
        return displacement_current_lidar, relative_pose, relative_matrix

    @staticmethod
    def transform_metric(points, transform):
        rotation = transform[..., :3, :3]
        translation = transform[..., :3, 3]
        output = torch.einsum('b...j,bij->b...i', points, rotation)
        view_shape = [translation.shape[0]] + [1] * (points.ndim - 2) + [3]
        return output + translation.reshape(view_shape)

    def warp_encoded(self, points, transform):
        metric_points = decode_points(points, self.pc_range)
        metric_points = self.transform_metric(metric_points, transform)
        return encode_points(metric_points, self.pc_range)

    def current_to_next(self, points, next_to_current):
        if self.frame_mode == 't0_aligned':
            return points
        return self.warp_encoded(points, self.inverse(next_to_current))

    def t0_to_next(self, points, next_to_t0):
        if self.frame_mode == 't0_aligned':
            return points
        return self.warp_encoded(points, self.inverse(next_to_t0))

    def build_relative_targets(self, ego2global_sequence):
        """Build adjacent ``T_next_to_current`` targets from ego poses."""
        current_to_global = ego2global_sequence[:, :-1]
        next_to_global = ego2global_sequence[:, 1:]
        global_to_current = self.inverse(current_to_global)
        next_to_current = torch.matmul(global_to_current, next_to_global)
        relative_pose = self.matrix_to_pose(next_to_current)
        planar_next_to_current = self.pose_to_matrix(relative_pose)
        return planar_next_to_current, relative_pose
