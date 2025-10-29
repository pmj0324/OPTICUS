"""
CNN-based models (ResNet) for scattering length prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * BasicBlock.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * BasicBlock.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet4(nn.Module):
    """
    ResNet with 4 layers for regression
    
    Args:
        block: Block type (default: BasicBlock)
        num_blocks: List of number of blocks per layer (default: [1,1,1,1])
        num_classes: Number of output classes (default: 1 for regression)
    """
    
    def __init__(self, block=BasicBlock, num_blocks=None, num_classes=1):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1, 1]
            
        self.in_planes = 64

        # Initial Convolution
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Average pooling and FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a layer with multiple blocks"""
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
            
        Returns:
            Output tensor of shape (B, 1)
        """
        # x: (batch, 1, 500, 500)
        out = F.relu(self.bn1(self.conv1(x)))  # (batch, 64, 250, 250)
        out = self.pool1(out)  # (batch, 64, 125, 125)

        out = self.layer1(out)  # (batch, 64, 125, 125)
        out = self.layer2(out)  # (batch, 128, 63, 63)
        out = self.layer3(out)  # (batch, 256, 32, 32)
        out = self.layer4(out)  # (batch, 512, 16, 16)

        out = self.avgpool(out)  # (batch, 512, 1, 1)
        out = torch.flatten(out, 1)  # (batch, 512)
        out = self.fc(out)  # (batch, 1)
        return out

