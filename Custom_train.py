import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import ArteryDataset
from custom_model import CustomArteryDetector
from torchvision import transforms

# Add custom dataset collate function to handle variable sized inputs
def collate_fn(batch):
    images = []
    target_locs = []
    target_conf = []
    
    for img, loc, conf in batch:
        # Make sure image is correctly formatted
        if isinstance(img, np.ndarray):
            # Convert numpy arrays to tensors
            if len(img.shape) == 2:  # Single channel grayscale
                img = torch.from_numpy(img).float().unsqueeze(0)  # Add channel dimension
            elif len(img.shape) == 3 and img.shape[2] == 1:  # Already has channel dimension
                img = torch.from_numpy(img).float().permute(2, 0, 1)
            elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB image
                img = torch.from_numpy(img).float().permute(2, 0, 1)
                # Convert to grayscale
                img = 0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]
        
        # Ensure image has correct dimensions and type
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Expected image to be torch.Tensor, got {type(img)}")
        
        if img.dim() == 2:
            img = img.unsqueeze(0)  # Add channel dimension
        
        # Normalize if needed
        if img.min() < 0 or img.max() > 1:
            img = img.float() / 255.0
        
        images.append(img)
        target_locs.append(loc)
        target_conf.append(conf)
    
    # Stack batches
    images = torch.stack(images)
    target_locs = torch.stack(target_locs)
    target_conf = torch.stack(target_conf)
    
    return images, target_locs, target_conf

class CustomArteryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss(reduction='none')
        self.loc_loss_weight = 1.0
    
    def forward(self, pred_locs, pred_conf, target_locs, target_conf):
        """
        Compute loss for artery detection
        pred_locs: [batch_size, 3] - predicted x, y, width
        pred_conf: [batch_size, 1] - predicted confidence
        target_locs: [batch_size, 3] - target x, y, width
        target_conf: [batch_size, 1] - target confidence
        """
        batch_size = pred_conf.size(0)
        
        # Compute confidence loss with class weighting
        pos_weight = 3.0  # Weight for positive samples
        sample_weights = torch.ones_like(target_conf)
        sample_weights[target_conf > 0.5] = pos_weight
        
        conf_loss = self.bce_loss(pred_conf, target_conf)
        conf_loss = (conf_loss * sample_weights).mean()
        
        # Compute location loss only for positive samples
        pos_mask = (target_conf > 0.5).float()
        
        # L1 loss for bounding box regression
        loc_loss = F.l1_loss(pred_locs, target_locs, reduction='none')
        loc_loss = (loc_loss * pos_mask.unsqueeze(-1)).sum() / (pos_mask.sum() + 1e-6)
        
        # Total loss
        total_loss = conf_loss + self.loc_loss_weight * loc_loss
        
        return total_loss, conf_loss, loc_loss

def train_custom_model(config):
    """Train the custom artery detection model"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = CustomArteryDetector().to(device)
    
    # Loss and optimizer
    criterion = CustomArteryLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    num_epochs = config.epochs
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_epochs/num_epochs,
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Create data loaders with augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        )
    ])
    
    train_dataset = ArteryDataset(
        config.data_path,
        split='train',
        annotation_type=config.annotation_type,
        transform=train_transform
    )
    
    val_dataset = ArteryDataset(
        config.data_path,
        split='val',
        annotation_type=config.annotation_type
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, target_locs, target_conf) in enumerate(progress_bar):
            images = images.to(device)
            target_locs = target_locs.to(device)
            target_conf = target_conf.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass
                pred_locs, pred_conf = model(images)
                
                # Compute loss
                loss, conf_loss, loc_loss = criterion(pred_locs, pred_conf, target_locs, target_conf)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                total_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'conf_loss': conf_loss.item(),
                    'loc_loss': loc_loss.item(),
                    'lr': optimizer.param_groups[0]['lr']
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, target_locs, target_conf in val_loader:
                images = images.to(device)
                target_locs = target_locs.to(device)
                target_conf = target_conf.to(device)
                
                try:
                    pred_locs, pred_conf = model(images)
                    loss, conf_loss, loc_loss = criterion(pred_locs, pred_conf, target_locs, target_conf)
                    val_loss += loss.item()
                except Exception as e:
                    print(f"Error in validation: {str(e)}")
                    continue
        
        val_loss /= len(val_loader)
        print(f"\nEpoch {epoch+1} - Avg train loss: {total_loss/len(train_loader):.4f}, Val loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"Saving best model with val_loss: {val_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_custom_model.pth')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("Training completed")
    
    return model 
