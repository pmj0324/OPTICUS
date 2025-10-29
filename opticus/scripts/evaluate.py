#!/usr/bin/env python3
"""
Evaluation script for OPTICUS models
"""

import argparse
import torch
from pathlib import Path

from opticus.models import ViT50_3block, ResNet4
from opticus.dataloader import load_h5_data, create_dataloaders
from opticus.utils import calculate_metrics
from opticus.utils.metrics import print_metrics
from opticus.analysis import save_all_plots
from opticus.configs import load_config


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate OPTICUS model")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to HDF5 data file (overrides config)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Which split to evaluate')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Experiment: {config.experiment.name}")
    
    # Override data path if provided
    if args.data is not None:
        config.data.hdf5_path = args.data
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {config.data.hdf5_path}...")
    images, labels, lbl_min, lbl_max = load_h5_data(config.data.hdf5_path)
    
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
    
    # Select dataloader
    if args.split == 'train':
        eval_loader = train_loader
    elif args.split == 'val':
        eval_loader = val_loader
    else:
        eval_loader = test_loader
    
    # Create model
    print(f"\nCreating model: {config.model.type}")
    model = create_model(config, device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    metrics = calculate_metrics(model, eval_loader, dataset, device)
    
    # Print metrics
    print_metrics(metrics, title=f"{args.split.upper()} SET METRICS")
    
    # Save plots
    if config.analysis.save_plots:
        plot_dir = Path(config.analysis.plot_dir) / args.split
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_all_plots(metrics, str(plot_dir), unit=config.analysis.unit)
        print(f"\nPlots saved to {plot_dir}/")


if __name__ == '__main__':
    main()

