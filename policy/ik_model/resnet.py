from torch.nn import SmoothL1Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNet50Pretrained(nn.Module):
    def __init__(self, output_dim=8):
        super(ResNet50Pretrained, self).__init__()
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
            nn.Linear(mlp_size, output_dim)  # Predicting two joint angles: q1 and q2
        )

    def forward(self, x):
        # Forward pass through ResNet-50
        x = self.resnet50(x)

        # Flatten the output from ResNet-50
        x = torch.flatten(x, 1)

        # Forward pass through the MLP layers
        x = self.mlp(x)

        return x
    def train_loss(self, goal, condition, loss_type="L1", weight: torch.Tensor = None):
        pred = self.forward(condition)

        if loss_type == "L1":
            loss = F.l1_loss(pred, goal, reduction="none")
        elif loss_type == "L2":
            loss = F.mse_loss(pred, goal, reduction="none")
        elif loss_type == "huber_loss":
            loss = F.smooth_l1_loss(pred, goal, reduction="none")
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        if weight is None:
            weight = torch.ones_like(loss).to(loss.device)

        loss = torch.mean(loss * weight)

        return loss
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If channels of input is not equal to output channel, then apply 1x1 conv to make them the same
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, input_dim):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        # x = self.fc(x)
        return x
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, input_dim, output_dim=1000):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64 * 4, 128, 4, stride=2)
        self.layer3 = self._make_layer(128 * 4, 256, 6, stride=2)
        self.layer4 = self._make_layer(256 * 4, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * 4, output_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride!= 1 or in_channels!= out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

        layers = []
        layers.append(BottleneckBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(out_channels * 4, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def train_loss(self, goal, condition, loss_type="L1", weight: torch.Tensor = None):
        pred = self.forward(condition)

        if loss_type == "L1":
            loss = F.l1_loss(pred, goal, reduction="none")
        elif loss_type == "L2":
            loss = F.mse_loss(pred, goal, reduction="none")
        elif loss_type == "huber_loss":
            loss = F.smooth_l1_loss(pred, goal, reduction="none")
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        if weight is None:
            weight = torch.ones_like(loss).to(loss.device)

        loss = torch.mean(loss * weight)

        return loss

# Test Code
# if __name__ == "__main__":

#     model = ResNet18(input_dim=6)

#     # print(sum([p.numel() for p in model.parameters()]))

#     input = torch.randn(1, 6, 128, 128)
#     output = model(input)
#     import ipdb; ipdb.set_trace()