import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomArteryDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder blocks with skip connections
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Vertical feature detector with different scales
        self.vertical_small = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.vertical_medium = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.vertical_large = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(9, 3), padding=(4, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Spatial attention modules
        self.attention1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.attention2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.attention3 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global context module
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Prediction heads
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.location_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # x, y, width
        )
    
    def forward(self, x):
        # Encoder path with vertical feature detection
        e1 = self.enc1(x)
        v1 = self.vertical_small(e1)
        a1 = self.attention1(v1)
        e1 = e1 * a1 + e1  # Residual connection
        
        e2 = self.enc2(e1)
        v2 = self.vertical_medium(e2)
        a2 = self.attention2(v2)
        e2 = e2 * a2 + e2  # Residual connection
        
        e3 = self.enc3(e2)
        v3 = self.vertical_large(e3)
        a3 = self.attention3(v3)
        e3 = e3 * a3 + e3  # Residual connection
        
        # Global context
        gc = self.global_context(e3)
        e3 = e3 * gc
        
        # Global pooling
        x = F.adaptive_avg_pool2d(e3, 1)
        x = x.view(x.size(0), -1)
        
        # Predict confidence and locations
        conf = self.confidence_head(x)
        loc = self.location_head(x)
        
        # Normalize locations to [0, 1] range
        loc = torch.sigmoid(loc)
        
        return loc, conf

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
