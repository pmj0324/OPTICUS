"""Evaluation metrics for OPTICUS."""
import numpy as np
import torch


def denormalize_labels(labels_norm, lbl_min, lbl_max):
    """Denormalize labels from [-1, 1] back to original scale.
    
    Args:
        labels_norm: Normalized labels in range [-1, 1]
        lbl_min: Original minimum label value
        lbl_max: Original maximum label value
        
    Returns:
        Denormalized labels in original scale
    """
    # [-1, 1] → [0, 1]
    labels_01 = (labels_norm + 1.0) / 2.0
    # [0, 1] → original scale
    labels_orig = labels_01 * (lbl_max - lbl_min) + lbl_min
    return labels_orig


def evaluate_model(model, test_loader, lbl_min, lbl_max, device):
    """Evaluate model on test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        lbl_min: Label minimum for denormalization
        lbl_max: Label maximum for denormalization
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing:
            - true_vals: Ground truth values
            - pred_vals: Predicted values
            - errors: Prediction errors (pred - true)
            - abs_rel_errors: Absolute relative errors in percentage
            - mse: Mean squared error
            - mae: Mean absolute error
            - mape: Mean absolute percentage error
    """
    model.eval()
    true_vals = []
    pred_vals = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # Denormalize to original scale
            labels_orig = denormalize_labels(
                labels.cpu().numpy(), lbl_min, lbl_max
            )
            preds_orig = denormalize_labels(
                outputs.cpu().numpy(), lbl_min, lbl_max
            )
            
            true_vals.extend(labels_orig.tolist())
            pred_vals.extend(preds_orig.tolist())
    
    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)
    
    # Calculate errors
    errors = pred_vals - true_vals
    abs_errors = np.abs(errors)
    abs_rel_errors = (abs_errors / np.abs(true_vals)) * 100.0  # percentage
    
    # Calculate metrics
    mse = np.mean(errors ** 2)
    mae = np.mean(abs_errors)
    mape = np.mean(abs_rel_errors)
    rmse = np.sqrt(mse)
    
    # Calculate percentiles
    perc_50 = np.percentile(abs_rel_errors, 50)
    perc_68 = np.percentile(abs_rel_errors, 68)
    perc_95 = np.percentile(abs_rel_errors, 95)
    
    results = {
        'true_vals': true_vals,
        'pred_vals': pred_vals,
        'errors': errors,
        'abs_errors': abs_errors,
        'abs_rel_errors': abs_rel_errors,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'percentile_50': perc_50,
        'percentile_68': perc_68,
        'percentile_95': perc_95
    }
    
    return results


def print_evaluation_summary(results):
    """Print evaluation metrics summary.
    
    Args:
        results: Dictionary returned by evaluate_model
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Number of samples: {len(results['true_vals'])}")
    print(f"\nRegression Metrics:")
    print(f"  MSE:  {results['mse']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  MAE:  {results['mae']:.6f}")
    print(f"  MAPE: {results['mape']:.2f}%")
    print(f"\nAbsolute Relative Error Percentiles:")
    print(f"  50th percentile: {results['percentile_50']:.2f}%")
    print(f"  68th percentile: {results['percentile_68']:.2f}%")
    print(f"  95th percentile: {results['percentile_95']:.2f}%")
    print("=" * 80 + "\n")

