import torch
import torch.nn as nn
from vit_pytorch import ViT
import torch.nn.functional as F
from utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation

class DETR_Encoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, ft_dim=128):
        super(DETR_Encoder, self).__init__()
    
        vit_encoder = ViT(
            image_size = img_size,
            patch_size = patch_size,
            num_classes = 1000,
            dim = ft_dim,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        self.encoder = nn.Sequential(
            vit_encoder.to_patch_embedding,
            vit_encoder.dropout,
            vit_encoder.transformer
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_dim = (img_size // patch_size)
        self.ft_dim = ft_dim
    
    def forward(self, img):
        out = self.encoder(img)     # 1 x num_patches x 64
        # print(out.shape)
        # sys.exit(1)
        out = out.permute(0, 2, 1)

        # Reshape and tessellate to the original image dimensions
        out = out.view(-1, self.ft_dim, self.patch_dim, self.patch_dim)

        out = torch.repeat_interleave(out, self.patch_size, dim=2)
        out = torch.repeat_interleave(out, self.patch_size, dim=3)

        return out

class PointNet_Encoder(nn.Module):
    def __init__(self, num_pc_feats, num_features):
        super(PointNet_Encoder, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], num_pc_feats, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 32, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(32, 1, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # print(l4_xyz.shape)
        # print(l4_points.shape)
        # print()

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = l0_points
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = self.conv3(x)
        
        return x
