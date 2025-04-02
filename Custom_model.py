import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomArteryDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature extraction layers - from scratch, no pretrained weights
        self.feature_extractor = nn.Sequential(
            # Initial convolution block
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),  # Using 1 channel for grayscale input
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Convolutional blocks with proper residual connections
        self.block1 = self._make_conv_block(32, 64)
        
        # Specialized vertical structure detector for arteries
        self.vert_detector = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.block2 = self._make_conv_block(64, 128)
        
        # Edge enhancement layer - important for artery boundaries
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.block3 = self._make_conv_block(128, 256)
        
        # Attention mechanism
        self.attention = self._make_attention_block(256)
        
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
    
    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block with proper residual connection"""
        return ResidualBlock(in_channels, out_channels)
    
    def _make_attention_block(self, channels):
        """Create an attention mechanism to focus on relevant features"""
        return nn.Sequential(
            # Spatial attention
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
            
            # Apply attention
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
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
        # Extract features
        x = self.feature_extractor(x)
        
        # Apply convolutional blocks with residual connections
        x = self.block1(x)
        x = self.vert_detector(x)
        x = self.block2(x)
        x = self.edge_enhance(x)
        x = self.block3(x)
        x = self.attention(x)
        features = self.pool(x)
        
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
