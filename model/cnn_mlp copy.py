import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Residual connection
        out = self.relu(out)
        return out

class CNNMLP(nn.Module):
    def __init__(self, out_size=7, base_channels=64):
        super().__init__()
        
        # Initial 3x3 conv layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3 residual blocks with 3x3 convolutions
        self.residual_blocks = nn.Sequential(
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        )
        
        # Global mean pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP layers (128, out_size)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_size)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        
        # Residual blocks
        x = self.residual_blocks(x)
        
        # Global mean pooling
        x = self.global_pool(x)
        
        # MLP layers
        x = self.mlp(x)
        
        return x
