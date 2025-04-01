import torch
import torch.nn as nn
import timm

class DepthViTMLP(nn.Module):
    def __init__(self, 
                 out_size=8,
                 pretrained=True,
                 model_name='vit_base_patch16_224',
                 mlp_size=256,
                 img_height=224,
                 img_width=224,
                 channels=4  
                 ):
        super().__init__()

        self.rgb_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=(img_height, img_width),
            in_chans=3,
            features_only=False
        )
        self.depth_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=(img_height, img_width),
            in_chans=1,
            features_only=False
        )

        dim = self.rgb_encoder.embed_dim  

        self.rgb_encoder.head = nn.Identity()
        self.depth_encoder.head = nn.Identity()

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=8),
            num_layers=2
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, out_size)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, img):
        # img: [B, 4, H, W]
        rgb_img = img[:, :3, :, :]        # [B, 3, H, W]
        depth_img = img[:, 3:, :, :]      # [B, 1, H, W]

        rgb_feat = self.rgb_encoder.forward_features(rgb_img)   # [B, N+1, C]
        depth_feat = self.depth_encoder.forward_features(depth_img)  # [B, N+1, C]

        fused_feat = rgb_feat + depth_feat  # [B, N+1, C]

        fused_feat = fused_feat.permute(1, 0, 2)  # [S, B, C] for Transformer
        decoded_feat = self.decoder(fused_feat)  # [S, B, C]
        decoded_feat = decoded_feat.permute(1, 0, 2)  # [B, S, C]

        cls_token = decoded_feat[:, 0]  # [B, C]

        out = self.mlp(cls_token)
        return out
