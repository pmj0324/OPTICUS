#!/usr/bin/env python3
"""
OPTICUS: Model Evaluation Example

This script demonstrates how to evaluate a trained OPTICUS model.
"""

import torch
from opticus.models import ViT50_3block, ResNet4
from opticus.dataloader import load_h5_data, create_dataloaders
from opticus.utils import calculate_metrics
from opticus.utils.metrics import print_metrics
from opticus.analysis import save_all_plots

def main():
    # Configuration
    config = {
        'data_path': '/path/to/your/data.h5',  # Update this path
        'checkpoint_path': '/path/to/your/checkpoint.pth',  # Update this path
        'model_type': 'ViT50_3block',  # or 'ResNet4'
        'batch_size': 32
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
    if config['model_type'] == 'ViT50_3block':
        model = ViT50_3block(
            img_size=500,
            patch_size=50,
            embed_dim=128,
            depth=3,
            num_heads=8,
            mlp_dim=512
        ).to(device)
    elif config['model_type'] == 'ResNet4':
        model = ResNet4(
            num_blocks=[1, 1, 1, 1],
            num_classes=1
        ).to(device)
    
    print(f"Model created: {config['model_type']}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {config['checkpoint_path']}")
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded state_dict-only checkpoint")
    
    model.eval()
    print("Model loaded successfully!")
    
    # Evaluate on all splits
    splits = {
        'train': train_loader,
        'validation': val_loader,
        'test': test_loader
    }
    
    all_metrics = {}
    
    for split_name, loader in splits.items():
        print(f"\nEvaluating on {split_name} set...")
        metrics = calculate_metrics(model, loader, dataset, device)
        all_metrics[split_name] = metrics
        print_metrics(metrics, title=f"{split_name.upper()} SET RESULTS")
    
    # Generate plots for test set
    test_metrics = all_metrics['test']
    save_all_plots(test_metrics, 'plots/evaluation_example', unit='cm')
    print("Plots saved to plots/evaluation_example/")
    
    # Performance summary
    print("\nModel Performance Summary:")
    print(f"{'Split':<12} {'MAE':<10} {'RMSE':<10} {'Mean Rel Error':<15} {'68th Percentile':<15}")
    print("-" * 70)
    
    for split_name, metrics in all_metrics.items():
        mae = metrics['mean_abs_error']
        rmse = metrics['rmse']
        mean_rel_error = metrics['mean_abs_rel_error']
        perc_68 = metrics['percentile_68']
        print(f"{split_name:<12} {mae:<10.6f} {rmse:<10.6f} {mean_rel_error:<15.3f} {perc_68:<15.3f}")

if __name__ == '__main__':
    main()
