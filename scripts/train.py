#!/usr/bin/env python3
"""
Training script for OPTICUS models
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from models import ViT50_3block, ResNet4
from training import load_h5_data, create_dataloaders
from utils import train_model
from configs import load_config


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(config, device):
    """Create model from config"""
    model_type = config.model.type
    
    if model_type == "ViT50_3block":
        model = ViT50_3block(
            img_size=config.model.img_size,
            patch_size=config.model.patch_size,
            embed_dim=config.model.embed_dim,
            depth=config.model.depth,
            num_heads=config.model.num_heads,
            mlp_dim=config.model.mlp_dim,
            num_classes=config.model.num_classes
        )
    elif model_type == "ResNet4":
        model = ResNet4(
            num_blocks=config.model.num_blocks,
            num_classes=config.model.num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def create_optimizer(model, config):
    """Create optimizer from config"""
    opt_type = config.training.optimizer.type
    
    if opt_type == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.optimizer.lr,
            weight_decay=config.training.optimizer.get('weight_decay', 0.0)
        )
    elif opt_type == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.optimizer.lr
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler from config"""
    sched_type = config.training.scheduler.type
    
    if sched_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.training.scheduler.get('mode', 'min'),
            factor=config.training.scheduler.get('factor', 0.5),
            patience=config.training.scheduler.get('patience', 4),
            verbose=config.training.scheduler.get('verbose', True)
        )
    elif sched_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.scheduler.get('T_max', 2),
            eta_min=config.training.scheduler.get('eta_min', 1e-5)
        )
    else:
        scheduler = None
    
    return scheduler


def main():
    parser = argparse.ArgumentParser(description="Train OPTICUS model")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to HDF5 data file (overrides config)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Experiment: {config.experiment.name}")
    
    # Override data path if provided
    if args.data is not None:
        config.data.hdf5_path = args.data
    
    # Set seed
    set_seed(config.experiment.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {config.data.hdf5_path}...")
    images, labels, lbl_min, lbl_max = load_h5_data(config.data.hdf5_path)
    print(f"Total samples: {len(images)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        images, labels, lbl_min, lbl_max,
        batch_size=config.data.batch_size,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.experiment.seed
    )
    
    # Create model
    print(f"\nCreating model: {config.model.type}")
    model = create_model(config, device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create criterion
    criterion = nn.MSELoss()
    
    # Create save directory
    save_dir = Path(config.checkpoint.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{config.experiment.name}_best.pth"
    
    # Train model
    print(f"\nStarting training...")
    print(f"Max epochs: {config.training.num_epochs}")
    print(f"Early stopping patience: {config.training.patience}")
    
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config.training.num_epochs,
        patience=config.training.patience,
        save_path=str(save_path),
        verbose=True
    )
    
    print(f"\nTraining complete! Best model saved to {save_path}")


if __name__ == '__main__':
    main()

