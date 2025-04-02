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
    def __init__(self, loc_weight=1.0, conf_weight=2.0, conf_threshold=0.5):
        super().__init__()
        self.loc_weight = loc_weight
        self.conf_weight = conf_weight
        self.conf_threshold = conf_threshold
        
    def forward(self, pred_locs, pred_conf, target_locs, target_conf):
        batch_size = pred_locs.size(0)
        
        # Binary classification loss with higher weight for positive samples
        pos_weight = torch.tensor([3.0]).to(pred_conf.device)
        conf_loss = F.binary_cross_entropy(pred_conf, target_conf, 
                                          pos_weight=pos_weight)
        
        # Only compute location loss for positive samples
        pos_mask = (target_conf > self.conf_threshold).float()
        
        # X coordinate loss (centered on artery)
        x_loss = F.smooth_l1_loss(
            pred_locs[:, 0] * pos_mask, 
            target_locs[:, 0] * pos_mask,
            reduction='sum'
        ) / (pos_mask.sum() + 1e-6)
        
        # Y coordinate loss (usually fixed at 0.5 for arteries)
        y_loss = F.smooth_l1_loss(
            pred_locs[:, 1] * pos_mask,
            target_locs[:, 1] * pos_mask,
            reduction='sum'
        ) / (pos_mask.sum() + 1e-6)
        
        # Width loss
        w_loss = F.smooth_l1_loss(
            pred_locs[:, 2] * pos_mask,
            target_locs[:, 2] * pos_mask,
            reduction='sum'
        ) / (pos_mask.sum() + 1e-6)
        
        # Height loss (usually fixed at 1.0 for arteries)
        h_loss = F.smooth_l1_loss(
            pred_locs[:, 3] * pos_mask,
            target_locs[:, 3] * pos_mask,
            reduction='sum'
        ) / (pos_mask.sum() + 1e-6)
        
        # Total location loss
        loc_loss = x_loss + y_loss + w_loss + h_loss
        
        # Final loss combining confidence and location components
        total_loss = self.conf_weight * conf_loss + self.loc_weight * loc_loss * pos_mask.mean()
        
        return total_loss, {
            'conf_loss': conf_loss.item(),
            'loc_loss': loc_loss.item(),
            'x_loss': x_loss.item(),
            'w_loss': w_loss.item()
        }

def train_custom_model(config):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = CustomArteryDetector().to(device)
    print("Created custom artery detector model from scratch")
    
    # Create datasets and dataloaders
    print(f"Loading datasets from {config.data_path}")
    train_dataset = ArteryDataset(
        config.data_path, 
        split='train',
        annotation_type=config.annotation_type,
        filter_empty=config.filter_empty
    )
    
    val_dataset = ArteryDataset(
        config.data_path, 
        split='val',
        annotation_type=config.annotation_type,
        filter_empty=config.filter_empty
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,  # Use custom collate function
        num_workers=0  # Use 0 workers for debugging
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        collate_fn=collate_fn,  # Use custom collate function
        num_workers=0  # Use 0 workers for debugging
    )
    
    # Loss and optimizer
    criterion = CustomArteryLoss(
        loc_weight=1.0,
        conf_weight=2.0
    )
    
    # Custom learning rate scheduling with warmup
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + np.cos((epoch - warmup_epochs) / (config.epochs - warmup_epochs) * np.pi))
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create directories for saving results
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    early_stop_patience = 10
    early_stop_counter = 0
    
    print(f"Starting training for {config.epochs} epochs")
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for images, target_locs, target_conf in progress_bar:
            try:
                # Move to device
                images = images.to(device)
                target_locs = target_locs.to(device)
                target_conf = target_conf.to(device)
                
                # Debug info
                if epoch == 0 and progress_bar.n == 0:
                    print(f"Input shape: {images.shape}, dtype: {images.dtype}")
                    print(f"Target locs shape: {target_locs.shape}, dtype: {target_locs.dtype}")
                    print(f"Target conf shape: {target_conf.shape}, dtype: {target_conf.dtype}")
                
                # Forward pass
                optimizer.zero_grad()
                pred_locs, pred_conf = model(images)
                
                # Compute loss
                loss, loss_components = criterion(pred_locs, pred_conf, target_locs, target_conf)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                # Update progress bar
                epoch_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'conf_loss': loss_components['conf_loss'],
                    'loc_loss': loss_components['loc_loss'],
                    'lr': optimizer.param_groups[0]['lr']
                })
            except Exception as e:
                print(f"Error in batch: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, target_locs, target_conf in val_loader:
                try:
                    images = images.to(device)
                    target_locs = target_locs.to(device)
                    target_conf = target_conf.to(device)
                    
                    pred_locs, pred_conf = model(images)
                    loss, _ = criterion(pred_locs, pred_conf, target_locs, target_conf)
                    val_loss += loss.item()
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, 'checkpoints/best_custom_model.pth')
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Save checkpoint every save_interval epochs
        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss
            }, f'checkpoints/custom_model_epoch_{epoch+1}.pth')
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('visualizations/loss_curve.png')
    
    # Save final model
    torch.save(model, 'checkpoints/final_custom_model.pth')
    print("Training complete! Final model saved.")
    
    return model 
