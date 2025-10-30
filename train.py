"""Main training script for OPTICUS.
OPTICUS 메인 학습 스크립트.
"""
import os
import argparse
import yaml
import torch

from models import ViT50_3block
from dataloader import create_dataloaders
from training import Trainer
from utils import set_seed, evaluate_model, print_evaluation_summary, plot_results, plot_training_history


def load_config(config_path):
    """Load configuration from YAML file.
    YAML 파일에서 설정 불러오기.
    
    Args / 인자:
        config_path: Path to YAML configuration file
                    YAML 설정 파일 경로
        
    Returns / 반환:
        Configuration dictionary
        설정 딕셔너리
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config, device):
    """Create model based on configuration.
    설정에 따라 모델 생성.
    
    Args / 인자:
        config: Configuration dictionary
               설정 딕셔너리
        device: Device to create model on
               모델을 생성할 디바이스
        
    Returns / 반환:
        Model instance
        모델 인스턴스
    """
    model_config = config['model']
    
    if model_config['name'] == 'ViT50_3block':
        model = ViT50_3block(
            img_size=model_config['img_size'],
            patch_size=model_config['patch_size'],
            embed_dim=model_config['embed_dim'],
            depth=model_config['depth'],
            num_heads=model_config['num_heads'],
            mlp_dim=model_config['mlp_dim'],
            num_classes=model_config['num_classes']
        )
    else:
        raise ValueError(f"Unknown model: {model_config['name']}")
    
    return model.to(device)


def main(args):
    """Main training function.
    메인 학습 함수.
    """
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.lr is not None:
        config['optimizer']['lr'] = args.lr
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['dataloader']['batch_size'] = args.batch_size
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories for checkpoints and results
    os.makedirs(os.path.dirname(config['training']['save_path']), exist_ok=True)
    if 'evaluation' in config:
        os.makedirs(os.path.dirname(config['evaluation']['save_results_plot']), exist_ok=True)
    
    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, lbl_min, lbl_max = create_dataloaders(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        test_path=config['data']['test_path'],
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        seed=config['seed']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Label range: [{lbl_min:.4f}, {lbl_max:.4f}]")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        history = trainer.resume_training(
            checkpoint_path=args.resume,
            additional_epochs=args.epochs,
            new_lr=args.lr,
            reset_scheduler=args.reset_scheduler
        )
    else:
        # Train model
        print("\nStarting training...")
        history = trainer.train()
    
    # Plot training history
    if 'evaluation' in config and config['evaluation'].get('save_history_plot'):
        plot_training_history(history, save_path=config['evaluation']['save_history_plot'])
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(config['training']['save_path'], map_location=device))
    results = evaluate_model(model, test_loader, lbl_min, lbl_max, device)
    
    # Print evaluation summary
    print_evaluation_summary(results)
    
    # Plot results
    if 'evaluation' in config and config['evaluation'].get('save_results_plot'):
        plot_results(results, save_path=config['evaluation']['save_results_plot'])
    else:
        plot_results(results)
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train OPTICUS model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--reset_scheduler', action='store_true',
                        help='Reset scheduler when resuming training')
    
    args = parser.parse_args()
    main(args)

