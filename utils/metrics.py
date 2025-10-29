"""
Metrics calculation utilities
"""

import numpy as np
import torch


def calculate_metrics(model, dataloader, dataset, device):
    """
    Calculate prediction metrics
    
    Args:
        model: Trained model
        dataloader: Dataloader to evaluate
        dataset: Dataset (for denormalization)
        device: Device to use
        
    Returns:
        dict: Dictionary containing metrics and predictions
    """
    model.eval()
    
    true_vals = []
    pred_vals = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Handle output shape
            if len(outputs.shape) > 1:
                outputs = outputs.squeeze(-1)
            
            # Denormalize to original scale
            labels_orig = dataset.denormalize_label(labels)
            preds_orig = dataset.denormalize_label(outputs)
            
            true_vals.extend(labels_orig.tolist() if isinstance(labels_orig, np.ndarray) 
                           else [labels_orig])
            pred_vals.extend(preds_orig.tolist() if isinstance(preds_orig, np.ndarray) 
                           else [preds_orig])
    
    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)
    
    # Calculate errors
    errors = pred_vals - true_vals
    abs_errors = np.abs(errors)
    abs_rel_errors = np.abs(errors / true_vals) * 100.0
    
    # Calculate metrics
    metrics = {
        'mean_error': np.mean(errors),
        'mean_abs_error': np.mean(abs_errors),
        'mean_abs_rel_error': np.mean(abs_rel_errors),
        'std_error': np.std(errors),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'percentile_68': np.percentile(abs_rel_errors, 68),
        'percentile_95': np.percentile(abs_rel_errors, 95),
        'true_vals': true_vals,
        'pred_vals': pred_vals,
        'errors': errors,
        'abs_rel_errors': abs_rel_errors
    }
    
    return metrics


def print_metrics(metrics, title="Metrics"):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        title: Title to print
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"Mean Error:                {metrics['mean_error']:.6f}")
    print(f"Mean Absolute Error:       {metrics['mean_abs_error']:.6f}")
    print(f"Mean Abs Relative Error:   {metrics['mean_abs_rel_error']:.3f}%")
    print(f"RMSE:                      {metrics['rmse']:.6f}")
    print(f"Std Error:                 {metrics['std_error']:.6f}")
    print(f"68th Percentile (Rel):     {metrics['percentile_68']:.3f}%")
    print(f"95th Percentile (Rel):     {metrics['percentile_95']:.3f}%")
    print(f"{'='*50}\n")

