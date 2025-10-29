#!/usr/bin/env python3
"""
OPTICUS: Simple ViT Training Example

This script demonstrates how to train a Vision Transformer model using OPTICUS.
"""

import torch
import torch.nn as nn
from opticus.models import ViT50_3block
from opticus.dataloader import load_h5_data, create_dataloaders
from opticus.utils import train_model, calculate_metrics
from opticus.utils.metrics import print_metrics
from opticus.analysis import save_all_plots

def main():
    # Configuration
    config = {
        'data_path': '/path/to/your/data.h5',  # Update this path
        'batch_size': 32,
        'num_epochs': 100,
        'patience': 10
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    images, labels, lbl_min, lbl_max = load_h5_data(config['data_path'])
    print(f"Total samples: {len(images)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        images, labels, lbl_min, lbl_max,
        batch_size=config['batch_size']
    )
    print("Data splits created successfully!")
    
    # Create model
    model = ViT50_3block(
        img_size=500,
        patch_size=50,
        embed_dim=128,
        depth=3,
        num_heads=8,
        mlp_dim=512
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, verbose=True
    )
    
    print("Training setup completed")
    
    # Train model
    print("Starting training...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config['num_epochs'],
        patience=config['patience'],
        save_path='checkpoints/vit_best.pth',
        verbose=True
    )
    print("Training completed!")
    
    # Evaluate model
    print("Evaluating model...")
    test_metrics = calculate_metrics(model, test_loader, dataset, device)
    print_metrics(test_metrics, title="TEST SET RESULTS")
    
    # Generate plots
    save_all_plots(test_metrics, 'plots/vit_example', unit='cm')
    print("Plots saved to plots/vit_example/")

if __name__ == '__main__':
    main()
