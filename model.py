"""
CNN Models for Strawberry Ripeness Classification
"""

import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation
from torchvision import models


class StrawberryCNN(nn.Module):
    """
    Custom CNN for Strawberry Ripeness Classification.
    
    Architecture:
        - 3 Convolutional blocks with BatchNorm and MaxPool
        - Adaptive pooling for flexible input sizes
        - 2 Fully connected layers
        
    Input: (batch_size, 3, 224, 224)
    Output: (batch_size, 4) - logits for 4 ripeness classes
    """
    
    def __init__(self, num_classes: int = 4):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            # Block 2: 32 -> 64 -> 128 channels
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            # Block 3: 128 -> 256 -> 512 channels
            Conv2dNormActivation(in_channels=128, out_channels=256, kernel_size=3),
            Conv2dNormActivation(in_channels=256, out_channels=256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            Conv2dNormActivation(in_channels=256, out_channels=512, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512 * 3 * 3, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class StrawberryResNet18(nn.Module):
    """
    ResNet18 fine-tuned for Strawberry Ripeness Classification.
    
    Uses pretrained ImageNet weights with modified final layer.
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.resnet18(weights=weights)
        
        # Replace final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True


def get_model(model_name: str = 'cnn', num_classes: int = 4, pretrained: bool = True):
    """
    Factory function to get model by name.
    
    Args:
        model_name: 'cnn' or 'resnet18'
        num_classes: Number of output classes
        pretrained: Use pretrained weights (for ResNet)
        
    Returns:
        PyTorch model
    """
    if model_name.lower() == 'cnn':
        return StrawberryCNN(num_classes=num_classes)
    elif model_name.lower() == 'resnet18':
        return StrawberryResNet18(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'cnn' or 'resnet18'")


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    from torchinfo import summary
    
    print("=" * 50)
    print("Custom CNN")
    print("=" * 50)
    cnn = StrawberryCNN()
    print(f"Parameters: {count_parameters(cnn):,}")
    summary(cnn, input_size=(1, 3, 224, 224))
    
    print("\n" + "=" * 50)
    print("ResNet18")
    print("=" * 50)
    resnet = StrawberryResNet18()
    print(f"Parameters: {count_parameters(resnet):,}")
