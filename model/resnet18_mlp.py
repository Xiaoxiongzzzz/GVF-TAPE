import torch
import torch.nn as nn
import torchvision

class ResNet18MLP(nn.Module):
    def __init__(self, out_size=9, pretrain_resnet_dir=None):
        super(ResNet18MLP, self).__init__()
        mlp_size = 256

        # Load a pre-trained ResNet-50 model
        self.resnet18 = torchvision.models.resnet18(pretrained=True)

        # Remove the fully connected layer of ResNet-50
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])

        # Define the MLP layers: 3 fully connected layers
        self.mlp = nn.Sequential(
            nn.Linear(512, mlp_size),  # ResNet-50 outputs a 2048-dimensional vector
            nn.ReLU(),
            nn.Linear(mlp_size, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, out_size) 
        )

    def forward(self, x):
        # Forward pass through ResNet-50
        x = self.resnet18(x)

        # Flatten the output from ResNet-50
        x = torch.flatten(x, 1)

        # Forward pass through the MLP layers
        x = self.mlp(x)

        return x