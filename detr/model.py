import torch
import torch.nn as nn
# from vit_pytorch import ViT
from vision_transformer_pytorch import VisionTransformer

class DETR_Encoder(nn.Module):
    def __init__(self, img_h=256, img_w=256, patch_size=16, ft_dim=128):
        super(DETR_Encoder, self).__init__()
    
        # vit_encoder = ViT(
        #     image_size = max(img_h, img_w),
        #     patch_size = patch_size,
        #     num_classes = 1000,
        #     dim = ft_dim,
        #     depth = 6,
        #     heads = 16,
        #     mlp_dim = 2048,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )
        vit = VisionTransformer.from_name('ViT-B_16', 
                                                    image_size = max(img_h, img_w), 
                                                    patch_size=patch_size,
                                                    emb_dim=ft_dim,
                                                    mlp_dim=3072)
        self.embedding = vit.embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, ft_dim))
        self.transformer = vit.transformer
        # self.encoder = nn.Sequential(
        #     vit_encoder.to_patch_embedding,
        #     vit_encoder.dropout,
        #     vit_encoder.transformer
        # )
        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.patch_dim_h = (img_h // patch_size)
        self.patch_dim_w = (img_w // patch_size)
        self.ft_dim = ft_dim
    
    def forward(self, img):
        emb = self.embedding(img)  # (n, c, gh, gw)
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        # prepend class token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)

        # transformer
        out = self.transformer(emb)
        # out = self.encoder(img)     # 1 x num_patches x 64
        
        # print(out.shape)
        # sys.exit(1)
        out = out[:, 1:, :]
        out = out.permute(0, 2, 1)

        # Reshape and tessellate to the original image dimensions
        out = out.view(-1, self.ft_dim, self.patch_dim_h, self.patch_dim_w)

        out = torch.repeat_interleave(out, self.patch_size, dim=2)
        out = torch.repeat_interleave(out, self.patch_size, dim=3)

        return out

if __name__ == '__main__':
    import torch
    model = DETR_Encoder(img_h=256, img_w=256, patch_size=8, ft_dim=64)

    # Test w/ B = 1, C = 3, H = 256, W = 256
    img = torch.rand(1, 3, 256, 256)
    out = model(img)
    print(out.shape)

    # Num params
    print(sum([p.data.nelement() for p in model.parameters()]))
    # print(p.shape)