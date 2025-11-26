import torch
import torch.nn as nn
import timm


class DepthCrossRGBViTMLP(nn.Module):
    def __init__(
        self,
        out_size=8,
        pretrained=True,
        model_name="vit_base_patch16_224",
        mlp_size=256,
        img_height=128,
        img_width=128,
        channels=4,
    ):
        super().__init__()

        self.rgb_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=(img_height, img_width),
            in_chans=3,
            features_only=False,
        )
        self.depth_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=(img_height, img_width),
            in_chans=1,
            features_only=False,
        )

        self.dim = self.rgb_encoder.embed_dim

        self.rgb_encoder.head = nn.Identity()
        self.depth_encoder.head = nn.Identity()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.dim, num_heads=8, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.dim, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, out_size),
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
        rgb_img = img[:, :3, :, :]
        depth_img = img[:, 3:, :, :]

        rgb_tokens = self.rgb_encoder.forward_features(rgb_img)  # [B, N+1, C]
        depth_tokens = self.depth_encoder.forward_features(depth_img)  # [B, N+1, C]

        depth_cls = depth_tokens[:, :1, :]  # [B, 1, C]

        rgb_all = rgb_tokens  # [B, N, C]

        # Cross-Attention: Query=RGB cls, Key/Value=Depth tokens
        cls_fused, _ = self.cross_attn(depth_cls, rgb_all, rgb_all)  # [B, 1, C]

        mlp_input = cls_fused.view(cls_fused.size(0), -1)

        x = self.mlp(mlp_input)

        return x
