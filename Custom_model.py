import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomArteryDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature extraction layers - from scratch, no pretrained weights
        self.feature_extractor = nn.Sequential(
            # Initial convolution block - maintain original channel size
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),  # Using 1 channel for grayscale input
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Simple convolutional architecture with controlled dimensionality
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Attention mechanism designed to work with 256 channels
        self.attention = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid()
        )
        
        # Final feature pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 4))
        
        # Confidence head - determines if artery is present
        self.confidence_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Location head - predicts bounding box coordinates
        self.location_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)  # x, y, w, h
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_attention_block(self, channels):
        """Create an attention mechanism to focus on relevant features"""
        return nn.Sequential(
            # Spatial attention
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def _initialize_weights(self):
        """Initialize weights with careful initialization for better convergence"""
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
        # Print input shape for debugging
        # print(f"Input shape: {x.shape}")
        
        # Extract features
        x = self.feature_extractor(x)
        # print(f"After feature_extractor: {x.shape}")
        
        # Apply convolutional blocks
        x = self.conv1(x)
        # print(f"After conv1: {x.shape}")
        
        x = self.conv2(x)
        # print(f"After conv2: {x.shape}")
        
        x = self.conv3(x)
        # print(f"After conv3: {x.shape}")
        
        # Apply attention mechanism
        attn = self.attention(x)
        x = x * attn
        # print(f"After attention: {x.shape}")
        
        # Global pooling
        features = self.pool(x)
        # print(f"After pooling: {features.shape}")
        
        # Get confidence score (artery vs non-artery)
        confidence = self.confidence_head(features)
        
        # Get location predictions
        raw_locations = self.location_head(features)
        
        # Transform location predictions to appropriate ranges
        locations = torch.cat([
            torch.sigmoid(raw_locations[:, 0:1]),  # x (normalized)
            torch.sigmoid(raw_locations[:, 1:2]),  # y (normalized)
            torch.sigmoid(raw_locations[:, 2:3]),  # width (normalized)
            torch.sigmoid(raw_locations[:, 3:4])   # height (normalized)
        ], dim=1)
        
        return locations, confidence

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path (for matching dimensions)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        
        # Pooling after the residual connection
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Main path
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut path
        residual = self.shortcut(residual)
        
        # Add residual connection
        out += residual
        out = self.relu(out)
        
        # Pooling after the addition
        out = self.pool(out)
        
        return out 
