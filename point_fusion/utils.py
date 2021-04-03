from functools import partial
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import Registry
# from mmdet3d.core.points import get_points_type
# from Structures_3D import CameraPoints, LiDARPoints
from Points import CameraPoints, LiDARPoints

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

VOXEL_ENCODERS = Registry('voxel_encoder')
MIDDLE_ENCODERS = Registry('middle_encoder')
FUSION_LAYERS = Registry('fusion_layer')

# def apply_3d_transformation(pcd, coords_type, img_meta, reverse=False):
#     """Apply transformation to input point cloud.
#     Args:
#         pcd (torch.Tensor): The point cloud to be transformed.
#         coords_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'
#         img_meta(dict): Meta info regarding data transformation.
#         reverse (bool): Reversed transformation or not.
#     Note:
#         The elements in img_meta['transformation_3d_flow']:
#         "T" stands for translation;
#         "S" stands for scale;
#         "R" stands for rotation;
#         "HF" stands for horizontal flip;
#         "VF" stands for vertical flip.
#     Returns:
#         torch.Tensor: The transformed point cloud.
#     """

#     dtype = pcd.dtype
#     device = pcd.device

#     pcd_rotate_mat = (
#         torch.tensor(img_meta['pcd_rotation'], dtype=dtype, device=device)
#         if 'pcd_rotation' in img_meta else torch.eye(
#             3, dtype=dtype, device=device))

#     pcd_scale_factor = (
#         img_meta['pcd_scale_factor'] if 'pcd_scale_factor' in img_meta else 1.)

#     pcd_trans_factor = (
#         torch.tensor(img_meta['pcd_trans'], dtype=dtype, device=device)
#         if 'pcd_trans' in img_meta else torch.zeros(
#             (3), dtype=dtype, device=device))

#     pcd_horizontal_flip = img_meta[
#         'pcd_horizontal_flip'] if 'pcd_horizontal_flip' in \
#         img_meta else False

#     pcd_vertical_flip = img_meta[
#         'pcd_vertical_flip'] if 'pcd_vertical_flip' in \
#         img_meta else False

#     flow = img_meta['transformation_3d_flow'] \
#         if 'transformation_3d_flow' in img_meta else []

#     pcd = pcd.clone()  # prevent inplace modification
#     pcd = get_points_type(coords_type)(pcd)

#     horizontal_flip_func = partial(pcd.flip, bev_direction='horizontal') \
#         if pcd_horizontal_flip else lambda: None
#     vertical_flip_func = partial(pcd.flip, bev_direction='vertical') \
#         if pcd_vertical_flip else lambda: None
#     if reverse:
#         scale_func = partial(pcd.scale, scale_factor=1.0 / pcd_scale_factor)
#         translate_func = partial(pcd.translate, trans_vector=-pcd_trans_factor)
#         # pcd_rotate_mat @ pcd_rotate_mat.inverse() is not
#         # exactly an identity matrix
#         # use angle to create the inverse rot matrix neither.
#         rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat.inverse())

#         # reverse the pipeline
#         flow = flow[::-1]
#     else:
#         scale_func = partial(pcd.scale, scale_factor=pcd_scale_factor)
#         translate_func = partial(pcd.translate, trans_vector=pcd_trans_factor)
#         rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat)

#     flow_mapping = {
#         'T': translate_func,
#         'S': scale_func,
#         'R': rotate_func,
#         'HF': horizontal_flip_func,
#         'VF': vertical_flip_func
#     }
#     for op in flow:
#         assert op in flow_mapping, f'This 3D data '\
#             f'transformation op ({op}) is not supported'
#         func = flow_mapping[op]
#         func()

#     return pcd.coord



def get_points_type(points_type):
    """Get the class of points according to coordinate type.
    Args:
        points_type (str): The type of points coordinate.
            The valid value are "CAMERA", "LIDAR", or "DEPTH".
    Returns:
        class: Points type.
    """
    if points_type == 'CAMERA':
        points_cls = CameraPoints
    elif points_type == 'LIDAR':
        points_cls = LiDARPoints
    else:
        raise ValueError('Only "points_type" of "CAMERA", "LIDAR", or "DEPTH"'
                         f' are supported, got {points_type}')

    return points_cls

def apply_3d_transformation(pcd, coords_type, img_meta, reverse=False):
    """Apply transformation to input point cloud.
    Args:
        pcd (torch.Tensor): The point cloud to be transformed.
        coords_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'
        img_meta(dict): Meta info regarding data transformation.
        reverse (bool): Reversed transformation or not.
    Note:
        The elements in img_meta['transformation_3d_flow']:
        "T" stands for translation;
        "S" stands for scale;
        "R" stands for rotation;
        "HF" stands for horizontal flip;
        "VF" stands for vertical flip.
    Returns:
        torch.Tensor: The transformed point cloud.
    """

    dtype = pcd.dtype
    device = pcd.device

    pcd_rotate_mat = (
        torch.tensor(img_meta['pcd_rotation'], dtype=dtype, device=device)
        if 'pcd_rotation' in img_meta else torch.eye(
            3, dtype=dtype, device=device))

    pcd_scale_factor = (
        img_meta['pcd_scale_factor'] if 'pcd_scale_factor' in img_meta else 1.)

    pcd_trans_factor = (
        torch.tensor(img_meta['pcd_trans'], dtype=dtype, device=device)
        if 'pcd_trans' in img_meta else torch.zeros(
            (3), dtype=dtype, device=device))

    pcd_horizontal_flip = img_meta[
        'pcd_horizontal_flip'] if 'pcd_horizontal_flip' in \
        img_meta else False

    pcd_vertical_flip = img_meta[
        'pcd_vertical_flip'] if 'pcd_vertical_flip' in \
        img_meta else False

    flow = img_meta['transformation_3d_flow'] \
        if 'transformation_3d_flow' in img_meta else []

    pcd = pcd.clone()  # prevent inplace modification
    pcd = get_points_type(coords_type)(pcd)

    horizontal_flip_func = partial(pcd.flip, bev_direction='horizontal') \
        if pcd_horizontal_flip else lambda: None
    vertical_flip_func = partial(pcd.flip, bev_direction='vertical') \
        if pcd_vertical_flip else lambda: None
    if reverse:
        scale_func = partial(pcd.scale, scale_factor=1.0 / pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=-pcd_trans_factor)
        # pcd_rotate_mat @ pcd_rotate_mat.inverse() is not
        # exactly an identity matrix
        # use angle to create the inverse rot matrix neither.
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat.inverse())

        # reverse the pipeline
        flow = flow[::-1]
    else:
        scale_func = partial(pcd.scale, scale_factor=pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=pcd_trans_factor)
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat)

    flow_mapping = {
        'T': translate_func,
        'S': scale_func,
        'R': rotate_func,
        'HF': horizontal_flip_func,
        'VF': vertical_flip_func
    }
    for op in flow:
        assert op in flow_mapping, f'This 3D data '\
            f'transformation op ({op}) is not supported'
        func = flow_mapping[op]
        func()

    return pcd.coord


# def rotation_3d_in_axis(points, angles, axis=0):
#     """Rotate points by angles according to axis.
#     Args:
#         points (torch.Tensor): Points of shape (N, M, 3).
#         angles (torch.Tensor): Vector of angles in shape (N,)
#         axis (int, optional): The axis to be rotated. Defaults to 0.
#     Raises:
#         ValueError: when the axis is not in range [0, 1, 2], it will \
#             raise value error.
#     Returns:
#         torch.Tensor: Rotated points in shape (N, M, 3)
#     """
#     rot_sin = torch.sin(angles)
#     rot_cos = torch.cos(angles)
#     ones = torch.ones_like(rot_cos)
#     zeros = torch.zeros_like(rot_cos)
#     if axis == 1:
#         rot_mat_T = torch.stack([
#             torch.stack([rot_cos, zeros, -rot_sin]),
#             torch.stack([zeros, ones, zeros]),
#             torch.stack([rot_sin, zeros, rot_cos])
#         ])
#     elif axis == 2 or axis == -1:
#         rot_mat_T = torch.stack([
#             torch.stack([rot_cos, -rot_sin, zeros]),
#             torch.stack([rot_sin, rot_cos, zeros]),
#             torch.stack([zeros, zeros, ones])
#         ])
#     elif axis == 0:
#         rot_mat_T = torch.stack([
#             torch.stack([zeros, rot_cos, -rot_sin]),
#             torch.stack([zeros, rot_sin, rot_cos]),
#             torch.stack([ones, zeros, zeros])
#         ])
#     else:
#         raise ValueError(f'axis should in range [0, 1, 2], got {axis}')

#     return torch.einsum('aij,jka->aik', (points, rot_mat_T))

# def limit_period(val, offset=0.5, period=np.pi):
#     """Limit the value into a period for periodic function.
#     Args:
#         val (torch.Tensor): The value to be converted.
#         offset (float, optional): Offset to set the value range. \
#             Defaults to 0.5.
#         period ([type], optional): Period of the value. Defaults to np.pi.
#     Returns:
#         torch.Tensor: Value in the range of \
#             [-offset * period, (1-offset) * period]
#     """
#     return val - torch.floor(val / period + offset) * period
