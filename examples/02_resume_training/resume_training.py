"""
Example 2: Resume Training from Checkpoint
기존 체크포인트에서 학습을 이어가는 예제
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import yaml
import torch
from models import ViT50_3block
from dataloader import create_dataloaders
from training import Trainer
from utils import set_seed, evaluate_model, print_evaluation_summary, plot_results, plot_training_history


def main():
    """체크포인트에서 학습을 이어가는 예제"""
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config_resume_training.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("Example 2: Resume Training from Checkpoint")
    print("=" * 80)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create directories
    os.makedirs(os.path.dirname(config['training']['save_path']), exist_ok=True)
    os.makedirs(os.path.dirname(config['evaluation']['save_results_plot']), exist_ok=True)
    
    # Load data
    print("\n[Step 1/6] Loading datasets...")
    train_loader, val_loader, test_loader, lbl_min, lbl_max = create_dataloaders(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        test_path=config['data']['test_path'],
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        seed=config['seed']
    )
    
    print(f"  ✓ Train samples: {len(train_loader.dataset)}")
    print(f"  ✓ Val samples: {len(val_loader.dataset)}")
    print(f"  ✓ Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\n[Step 2/6] Creating model...")
    model = ViT50_3block(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        mlp_dim=config['model']['mlp_dim'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Total parameters: {total_params:,}")
    
    # Load checkpoint
    print(f"\n[Step 3/6] Loading checkpoint from: {config['checkpoint']['load_path']}")
    checkpoint_path = config['checkpoint']['load_path']
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_path)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"  ✓ Checkpoint loaded successfully")
    
    # Create trainer
    print(f"\n[Step 4/6] Resuming training for {config['training']['num_epochs']} additional epochs...")
    print(f"  ✓ New learning rate: {config['optimizer']['lr']:.6g}")
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Reset early stopping (starting fresh)
    trainer.best_val_loss = float('inf')
    trainer.wait = 0
    
    # Train
    history = trainer.train()
    
    # Plot training history
    print("\n[Step 5/6] Plotting training history...")
    plot_training_history(history, save_path=config['evaluation']['save_history_plot'])
    print(f"  ✓ Training history saved to {config['evaluation']['save_history_plot']}")
    
    # Evaluate on test set
    print("\n[Step 6/6] Evaluating on test set...")
    model.load_state_dict(torch.load(config['training']['save_path'], map_location=device))
    results = evaluate_model(model, test_loader, lbl_min, lbl_max, device)
    
    # Print evaluation summary
    print_evaluation_summary(results)
    
    # Plot results
    plot_results(results, save_path=config['evaluation']['save_results_plot'])
    print(f"  ✓ Evaluation results saved to {config['evaluation']['save_results_plot']}")
    
    print("\n" + "=" * 80)
    print("Resume training completed successfully!")
    print(f"Resumed model saved to: {config['training']['save_path']}")
    print("=" * 80)


if __name__ == '__main__':
    main()

