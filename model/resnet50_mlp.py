import torch
import torch.nn as nn
import torchvision


class ResNet50MLP(nn.Module):
    def __init__(self, out_size=9, pretrain_resnet_dir=None):
        super(ResNet50MLP, self).__init__()
        mlp_size = 256

        # Load a pre-trained ResNet-50 model
        self.resnet50 = torchvision.models.resnet50(pretrained=True)

        # Remove the fully connected layer of ResNet-50
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        # Define the MLP layers: 3 fully connected layers
        self.mlp = nn.Sequential(
            nn.Linear(2048, mlp_size),  # ResNet-50 outputs a 2048-dimensional vector
            nn.ReLU(),
            nn.Linear(mlp_size, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, out_size),  # Predicting two joint angles: q1 and q2
        )

    def forward(self, x):
        # Forward pass through ResNet-50
        x = self.resnet50(x)

        # Flatten the output from ResNet-50
        x = torch.flatten(x, 1)

        # Forward pass through the MLP layers
        x = self.mlp(x)

        return x
