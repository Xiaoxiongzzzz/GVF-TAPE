import torch
from resnet50_mlp import ResNet50MLP
from cnn_mlp import CNNMLP
from vit_mlp import ViTMLP
from resnet18_mlp import ResNet18MLP


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compare_models(image_size=224, out_size=8):
    # Initialize models
    resnet = ResNet50MLP(out_size=out_size)
    cnn = CNNMLP(out_size=out_size)
    vit = ViTMLP(out_size=out_size, pretrained=True)
    resnet18 = ResNet18MLP(out_size=out_size)
    
    # Create dummy input
    x = torch.randn(1, 3, image_size, image_size)
    
    # Count parameters
    resnet_params = count_parameters(resnet)
    cnn_params = count_parameters(cnn)
    vit_params = count_parameters(vit)
    resnet18_params = count_parameters(resnet18)

    print("Model Parameter Comparison:")
    print("-" * 50)
    print(f"ResNet18MLP: {resnet18_params/1e6:.2f}M parameters")
    print(f"ResNet50MLP: {resnet_params/1e6:.2f}M parameters")
    print(f"CNNMLP:      {cnn_params/1e6:.2f}M parameters")
    print(f"ViTMLP:      {vit_params/1e6:.2f}M parameters")
    print("-" * 50)
    
    # Optional: Break down by components
    resnet50_backbone = sum(p.numel() for p in resnet.resnet50.parameters() if p.requires_grad)
    resnet50_mlp = sum(p.numel() for p in resnet.mlp.parameters() if p.requires_grad)
    
    resnet18_backbone = sum(p.numel() for p in resnet18.resnet18.parameters() if p.requires_grad)
    resnet18_mlp = sum(p.numel() for p in resnet18.mlp.parameters() if p.requires_grad)
    
    vit_backbone = sum(p.numel() for p in vit.vit.parameters() if p.requires_grad)
    vit_mlp = sum(p.numel() for p in vit.mlp.parameters() if p.requires_grad)
    
    print("\nBreakdown:")
    print("ResNet50MLP:")
    print(f"  - Backbone: {resnet50_backbone/1e6:.2f}M")
    print(f"  - MLP:      {resnet50_mlp/1e6:.2f}M")
    print("\nResNet18MLP:")
    print(f"  - Backbone: {resnet18_backbone/1e6:.2f}M")
    print(f"  - MLP:      {resnet18_mlp/1e6:.2f}M")
    print("\nViTMLP:")
    print(f"  - Backbone: {vit_backbone/1e6:.2f}M")
    print(f"  - MLP:      {vit_mlp/1e6:.2f}M")

if __name__ == "__main__":
    compare_models()