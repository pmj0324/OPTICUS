"""
Example 3: Evaluation and Visualization
학습된 모델을 불러와서 평가하고 결과를 시각화하는 예제
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import yaml
import torch
import pandas as pd
from models import ViT50_3block
from dataloader import create_dataloaders
from utils import set_seed, evaluate_model, print_evaluation_summary, plot_results


def main():
    """학습된 모델을 평가하고 결과를 시각화하는 예제"""
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config_evaluation.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("Example 3: Evaluation and Visualization")
    print("=" * 80)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create directories
    os.makedirs(os.path.dirname(config['evaluation']['save_results_plot']), exist_ok=True)
    
    # Load data
    print("\n[Step 1/4] Loading datasets...")
    train_loader, val_loader, test_loader, lbl_min, lbl_max = create_dataloaders(
        train_path=config['data']['train_path'],
        val_path=config['data']['val_path'],
        test_path=config['data']['test_path'],
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        seed=config['seed']
    )
    
    # Select loader based on config
    dataset_choice = config['evaluation']['dataset']
    if dataset_choice == 'train':
        eval_loader = train_loader
        dataset_name = 'Training'
    elif dataset_choice == 'val':
        eval_loader = val_loader
        dataset_name = 'Validation'
    else:
        eval_loader = test_loader
        dataset_name = 'Test'
    
    print(f"  ✓ Evaluating on {dataset_name} set ({len(eval_loader.dataset)} samples)")
    print(f"  ✓ Label range: [{lbl_min:.4f}, {lbl_max:.4f}]")
    
    # Create model
    print("\n[Step 2/4] Creating model...")
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
    print(f"\n[Step 3/4] Loading checkpoint from: {config['checkpoint']['load_path']}")
    checkpoint_path = config['checkpoint']['load_path']
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_path)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"  ✓ Checkpoint loaded successfully")
    
    # Evaluate
    print(f"\n[Step 4/4] Evaluating on {dataset_name} set...")
    results = evaluate_model(model, eval_loader, lbl_min, lbl_max, device)
    
    # Print evaluation summary
    print_evaluation_summary(results)
    
    # Plot results
    print("\nPlotting results...")
    plot_results(results, save_path=config['evaluation']['save_results_plot'])
    print(f"  ✓ Evaluation plot saved to {config['evaluation']['save_results_plot']}")
    
    # Save predictions to CSV (optional)
    if config['evaluation'].get('save_predictions_csv'):
        csv_path = config['evaluation']['save_predictions_csv']
        df = pd.DataFrame({
            'true_values': results['true_vals'],
            'predicted_values': results['pred_vals'],
            'errors': results['errors'],
            'absolute_errors': results['abs_errors'],
            'absolute_relative_errors_percent': results['abs_rel_errors']
        })
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Predictions saved to {csv_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()

