"""
Training utilities
"""

import torch


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        
    Returns:
        Average training loss
    """
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle output shape (squeeze if needed)
        if len(outputs.shape) > 1:
            outputs = outputs.squeeze(-1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    avg_loss = running_loss / len(train_loader.dataset)
    return avg_loss


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate for one epoch
    
    Args:
        model: Model to validate
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to use
        
    Returns:
        Average validation loss
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Handle output shape
            if len(outputs.shape) > 1:
                outputs = outputs.squeeze(-1)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    
    avg_loss = running_loss / len(val_loader.dataset)
    return avg_loss


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=100,
    patience=10,
    save_path='best_model.pth',
    verbose=True
):
    """
    Complete training loop with early stopping
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        save_path: Path to save best model
        verbose: Whether to print progress
        
    Returns:
        Trained model
    """
    best_val_loss = float('inf')
    wait = 0
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler is not None:
            if hasattr(scheduler, 'step'):
                # For ReduceLROnPlateau
                if 'ReduceLROnPlateau' in scheduler.__class__.__name__:
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        if verbose:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.6g}")
        
        # Early stopping and best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
            if verbose:
                print(f"  → New best val loss: {best_val_loss:.6f}, saved to {save_path}")
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

