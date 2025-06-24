# In train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model import SiameseTransformer
from data_loader import TripletSignatureDataset

# --- Hyperparameters and Configuration ---
TRAIN_DIR = './data/train'
VAL_DIR = './data/val'
MODEL_SAVE_PATH = './trained_models/siamese_transformer.pth'

MODEL_CONFIG = {
    'image_size': 224,
    'patch_size': 16,
    'in_channels': 1,
    'embed_dim': 512,      # TODO: review
    'depth': 8,            # TODO: review
    'heads': 8,            # TODO: review
    'mlp_dim': 2048        # TODO: review
}

TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 3e-4,   # Initial LR. TODO: Review
    'weight_decay': 1e-4,    # Regularization
    'margin': 0.5,           # TODO: May need to review different margins
    'warmup_epochs': 10,
    'patience': 15           # Early stopping
}

# Improved training loop
def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch):
    """
    Improved training function for one epoch with better metrics and gradient handling.
    """
    model.train()
    total_loss = 0.0
    correct_triplets = 0
    total_triplets = 0
    
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    
    for batch_idx, (anchor, positive, negative) in enumerate(progress_bar):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        
        # Get embeddings
        anchor_emb = model.backbone(anchor)
        positive_emb = model.backbone(positive)
        negative_emb = model.backbone(negative)
        
        # Compute loss
        loss = loss_fn(anchor_emb, positive_emb, negative_emb)
        
        # Check triplet accuracy
        with torch.no_grad():
            pos_dist = torch.norm(anchor_emb - positive_emb, dim=1)
            neg_dist = torch.norm(anchor_emb - negative_emb, dim=1)
            correct_triplets += (pos_dist < neg_dist).sum().item()
            total_triplets += anchor.size(0)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Update progress bar
        current_accuracy = correct_triplets / total_triplets if total_triplets > 0 else 0
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_accuracy:.3f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # Learning rate warmup
        if epoch < TRAINING_CONFIG['warmup_epochs']:
            lr_scale = min(1.0, float(batch_idx + 1) / len(train_loader))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * TRAINING_CONFIG['learning_rate']
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_triplets / total_triplets
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, loss_fn, device):
    """
    Validation function for one epoch.
    """
    model.eval()
    total_loss = 0.0
    correct_triplets = 0
    total_triplets = 0
    
    with torch.no_grad():
        for anchor, positive, negative in tqdm(val_loader, desc="Validating"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_emb = model.backbone(anchor)
            positive_emb = model.backbone(positive)
            negative_emb = model.backbone(negative)
            
            loss = loss_fn(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()
            
            # Calculate accuracy
            pos_dist = torch.norm(anchor_emb - positive_emb, dim=1)
            neg_dist = torch.norm(anchor_emb - negative_emb, dim=1)
            correct_triplets += (pos_dist < neg_dist).sum().item()
            total_triplets += anchor.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_triplets / total_triplets
    
    return avg_loss, accuracy


def main():
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading datasets...")
    train_dataset = TripletSignatureDataset(data_dir=TRAIN_DIR)
    val_dataset = TripletSignatureDataset(data_dir=VAL_DIR)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    print("Datasets loaded.")

    # --- Model, Loss, and Optimizer ---
    # --- Basic model structure ---
    # model = SiameseTransformer(
    #     image_size=224,
    #     patch_size=16,
    #     in_channels=1,
    #     embed_dim=256,
    #     depth=4,
    #     heads=4,
    #     mlp_dim=1024
    # ).to(device)

    model = SiameseTransformer(**MODEL_CONFIG).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    loss_fn = nn.TripletMarginLoss(margin=TRAINING_CONFIG['margin'], p=2) # p=2 for Euclidean distance
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )

    # Learning rate scheduler (starts after warmup)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=TRAINING_CONFIG['epochs'] - TRAINING_CONFIG['warmup_epochs']
    )

    # Training tracking
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Create model save directory
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # --- Main Training Loop ---
    print(f"\nStarting training for {TRAINING_CONFIG['epochs']} epochs...")
    print(f"Warmup for first {TRAINING_CONFIG['warmup_epochs']} epochs")
    
    for epoch in range(TRAINING_CONFIG['epochs']):
        print(f"\n--- Epoch {epoch+1}/{TRAINING_CONFIG['epochs']} ---")
        
        # Training phase - using the new train_epoch function
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, loss_fn, device)
        
        # Step scheduler after warmup period
        if epoch >= TRAINING_CONFIG['warmup_epochs']:
            scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'model_config': MODEL_CONFIG,
                'training_config': TRAINING_CONFIG
            }, MODEL_SAVE_PATH)
            
            print(f"✓ Model improved and saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{TRAINING_CONFIG['patience']}")
            
            if patience_counter >= TRAINING_CONFIG['patience']:
                print("Early stopping triggered!")
                break

    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

    # --- Basic Training Loop ---
    # for epoch in range(EPOCHS):
    #     print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
    #     # Training phase
    #     model.train()
    #     train_loss = 0.0
    #     for anchor, positive, negative in tqdm(train_loader, desc="Training"):
    #         anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

    #         optimizer.zero_grad()
            
    #         # Get embeddings from the model's backbone
    #         anchor_emb = model.backbone(anchor)
    #         positive_emb = model.backbone(positive)
    #         negative_emb = model.backbone(negative)
            
    #         loss = loss_fn(anchor_emb, positive_emb, negative_emb)
    #         loss.backward()
    #         optimizer.step()
            
    #         train_loss += loss.item()
        
    #     avg_train_loss = train_loss / len(train_loader)
    #     print(f"Average Training Loss: {avg_train_loss:.4f}")

    #     # Validation phase
    #     model.eval()
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         for anchor, positive, negative in tqdm(val_loader, desc="Validating"):
    #             anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                
    #             anchor_emb = model.backbone(anchor)
    #             positive_emb = model.backbone(positive)
    #             negative_emb = model.backbone(negative)
                
    #             loss = loss_fn(anchor_emb, positive_emb, negative_emb)
    #             val_loss += loss.item()
        
    #     avg_val_loss = val_loss / len(val_loader)
    #     print(f"Average Validation Loss: {avg_val_loss:.4f}")

    #     # Save the best model
    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         torch.save(model.state_dict(), MODEL_SAVE_PATH)
    #         print(f"Model improved and saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()