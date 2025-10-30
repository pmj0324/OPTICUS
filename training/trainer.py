"""Training utilities for OPTICUS."""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


class Trainer:
    """Trainer class for model training and validation.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to train on (cuda/cpu)
    """
    
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup criterion
        self.criterion = nn.MSELoss()
        
        # Setup optimizer
        optimizer_config = config['optimizer']
        self.optimizer = AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
        
        # Setup scheduler
        scheduler_config = config['scheduler']
        if scheduler_config['type'] == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config.get('mode', 'min'),
                factor=scheduler_config.get('factor', 0.7),
                patience=scheduler_config.get('patience', 2),
                threshold=scheduler_config.get('threshold', 0.0),
                threshold_mode=scheduler_config.get('threshold_mode', 'abs'),
                verbose=scheduler_config.get('verbose', True)
            )
            self.scheduler_type = 'plateau'
        elif scheduler_config['type'] == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 100),
                eta_min=scheduler_config.get('eta_min', 1e-7)
            )
            self.scheduler_type = 'cosine'
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_config['type']}")
        
        # Training settings
        self.num_epochs = config['training']['num_epochs']
        self.patience = config['training'].get('patience', 10)
        self.save_path = config['training']['save_path']
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.wait = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_train_loss = 0.0
        
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
        
        train_loss = running_train_loss / len(self.train_loader.dataset)
        return train_loss
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
        
        val_loss = running_val_loss / len(self.val_loader.dataset)
        return val_loss
    
    def train(self):
        """Train the model for multiple epochs with early stopping."""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 80)
        
        for epoch in range(1, self.num_epochs + 1):
            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            # Get current learning rate
            lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)
            
            # Print progress
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | LR: {lr:.6g}")
            
            # Update scheduler
            if self.scheduler_type == 'plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.wait = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  → New best val loss: {self.best_val_loss:.6f}, "
                      f"saved to {self.save_path}")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print("-" * 80)
        print(f"Training completed. Best validation loss: {self.best_val_loss:.6f}")
        return self.history
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model weights from {checkpoint_path}")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"Loaded model state_dict from {checkpoint_path}")
    
    def resume_training(self, checkpoint_path, additional_epochs=None, 
                       new_lr=None, reset_scheduler=True):
        """Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            additional_epochs: Number of additional epochs (overrides config if provided)
            new_lr: New learning rate (overrides config if provided)
            reset_scheduler: Whether to reset the scheduler
        """
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Update training parameters if provided
        if additional_epochs is not None:
            self.num_epochs = additional_epochs
        if new_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        # Reset scheduler if requested
        if reset_scheduler:
            scheduler_config = self.config['scheduler']
            if scheduler_config['type'] == 'ReduceLROnPlateau':
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode=scheduler_config.get('mode', 'min'),
                    factor=scheduler_config.get('factor', 0.7),
                    patience=scheduler_config.get('patience', 2),
                    threshold=scheduler_config.get('threshold', 0.0),
                    threshold_mode=scheduler_config.get('threshold_mode', 'abs'),
                    verbose=scheduler_config.get('verbose', True)
                )
            elif scheduler_config['type'] == 'CosineAnnealingLR':
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config.get('T_max', 100),
                    eta_min=scheduler_config.get('eta_min', 1e-7)
                )
        
        # Reset early stopping
        self.best_val_loss = float('inf')
        self.wait = 0
        
        print(f"Resuming training for {self.num_epochs} additional epochs...")
        print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6g}")
        
        # Continue training
        return self.train()

