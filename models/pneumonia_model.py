import torch
import torch.nn as nn
from torchvision import models

class PneumoniaDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(PneumoniaDetector, self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Modify first layer for 1-channel grayscale input
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, original_conv1.out_channels, 
                                        kernel_size=original_conv1.kernel_size,
                                        stride=original_conv1.stride,
                                        padding=original_conv1.padding,
                                        bias=original_conv1.bias)
        
        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))
        
        # Replace final layer for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)