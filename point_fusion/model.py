import os
import sys

import torch
import torch.nn as nn

from cores import DETR_Encoder, PointNet_Encoder
from fusion import PointFusion

if __name__ == '__main__':
    batch_size = 2
    # Input RGB
    in_rgb = torch.randn(2, 3, 64, 64)

    # Input point cloud
    num_points = 1024
    num_pc_feats = 6
    in_pc = torch.randn(2, num_pc_feats, num_points)

    # Make backbones
    ft_dim = 128
    img_size = 64
    rgb_encoder = DETR_Encoder(img_size=img_size, patch_size=8, ft_dim=ft_dim)
    pc_encoder = PointNet_Encoder(num_pc_feats)
    
    # Run backbones
    rgb_featurized = rgb_encoder(in_rgb)
    print(rgb_featurized.shape)
    print()
    pc_featurized = pc_encoder(in_pc)
    print(pc_featurized.shape)
    print()

    # Make point fusion core
    img_channels = ft_dim
    pts_channels = num_pc_feats - 3
    mid_channels = 64
    out_channels = 128

    lidar2img = torch.tensor(
        [[6.0294e+02, -7.0791e+02, -1.2275e+01, -1.7094e+02],
         [1.7678e+02, 8.8088e+00, -7.0794e+02, -1.0257e+02],
         [9.9998e-01, -1.5283e-03, -5.2907e-03, -3.2757e-01],
         [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]])

    #  all use default
    img_meta = {
        'transformation_3d_flow': ['R', 'S', 'T', 'HF'],
        'input_shape': [img_size, img_size],
        'img_shape': [img_size, img_size],
        'lidar2img': lidar2img,
    }

    fuse = PointFusion(img_channels, pts_channels, mid_channels, out_channels, 
                        img_levels=[0], lateral_conv=True,
                        align_corners=False,
                        activate_out=True,
                        fuse_out=False)

    # Obtain fused point cloud representation
    pts = in_pc[:, :3, :].transpose(1, 2)
    # pts = in_pc[:, :3, :]
    pts_feats = in_pc[:, 3:, :].transpose(1, 2)
    # pts_feats = in_pc[:, 3:, :]
    out = fuse.forward([rgb_featurized], pts, pts_feats, [img_meta, img_meta])
    print(out.shape)
        # out = out.reshape(batch_size, num_points, ft_dim).transpose(1, 2)
        # print(out.shape)

