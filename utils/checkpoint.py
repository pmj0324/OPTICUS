"""
Checkpoint utilities for saving and loading models
"""

import torch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    val_loss,
    save_path
):
    """
    Save a complete checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")


def load_checkpoint(
    model,
    checkpoint_path,
    optimizer=None,
    scheduler=None,
    device='cpu'
):
    """
    Load a checkpoint
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint to
        
    Returns:
        tuple: (model, optimizer, scheduler, epoch, train_loss, val_loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if it's a full checkpoint or just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        train_loss = checkpoint.get('train_loss', None)
        val_loss = checkpoint.get('val_loss', None)
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {epoch}")
        return model, optimizer, scheduler, epoch, train_loss, val_loss
    else:
        # Just a state_dict
        model.load_state_dict(checkpoint)
        print("Loaded state_dict-only checkpoint")
        return model, optimizer, scheduler, 0, None, None

