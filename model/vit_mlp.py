import torch
import torch.nn as nn
import timm
from einops import repeat

class ViTMLP(nn.Module):
    def __init__(self, 
                 out_size=8,
                 pretrained=True,
                 model_name='vit_base_patch16_224',  # or 'vit_large_patch16_224'
                 mlp_size=256,
                 img_height=128,
                 img_width=128
                 ):
        super().__init__()
        
        # Create model with custom image size
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=(img_height, img_width),
        )
        
        # Get embedding dimension from the model
        dim = self.vit.embed_dim  # 768 for base, 1024 for large
        
        # Remove the original head
        self.vit.head = nn.Identity()
        
        # Use exactly the same MLP as ResNet50_MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_size),  # ViT outputs embed_dim-dimensional vector
            nn.ReLU(),
            nn.Linear(mlp_size, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, out_size)  # Final output
        )
        
        # Initialize MLP weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, img):
        # Get CLS token embedding from ViT
        x = self.vit(img)
        # Pass through MLP head
        x = self.mlp(x)
        return x
