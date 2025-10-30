"""Visualization utilities for OPTICUS."""
import numpy as np
import matplotlib.pyplot as plt


def plot_results(results, save_path=None):
    """Plot evaluation results with 4 subplots.
    
    Args:
        results: Dictionary returned by evaluate_model containing:
            - true_vals: Ground truth values
            - pred_vals: Predicted values
            - errors: Prediction errors
            - abs_rel_errors: Absolute relative errors in percentage
        save_path: Optional path to save the figure
    """
    true_vals = results['true_vals']
    pred_vals = results['pred_vals']
    errors = results['errors']
    abs_rel_errors = results['abs_rel_errors']
    
    # Compute plot ranges
    mn = min(true_vals.min(), pred_vals.min())
    mx = max(true_vals.max(), pred_vals.max())
    unique_true_labels = np.unique(true_vals)
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1) Predicted vs True with ±5% Band
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(true_vals, pred_vals, s=10, color='black', label='Predictions')
    ax1.plot([mn, mx], [mn, mx], color='red', label='y = x')
    ax1.plot([mn, mx], [0.95*mn, 0.95*mx], '--', color='blue', label='y = 0.95x')
    ax1.plot([mn, mx], [1.05*mn, 1.05*mx], '--', color='blue', label='y = 1.05x')
    ax1.fill_between([mn, mx], [0.95*mn, 0.95*mx], [1.05*mn, 1.05*mx], 
                      color='blue', alpha=0.2)
    ax1.set_xticks(unique_true_labels)
    ax1.set_xticklabels([f"{val:.2f}" for val in unique_true_labels], rotation=45)
    ax1.set_title("Predicted vs True Scattering Length (±5% Band)")
    ax1.set_xlabel("True Scattering Length (cm)")
    ax1.set_ylabel("Predicted Scattering Length (cm)")
    ax1.set_xlim(mn, mx)
    ax1.set_ylim(mn * 0.9, mx * 1.1)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2) Error vs True with ±5% Band
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(true_vals, errors, s=10, color='black', label='Prediction Errors')
    ax2.plot([mn, mx], [0, 0], '-', color='red', label='No Error')
    ax2.plot([mn, mx], [-0.05*mn, -0.05*mx], '--', color='blue', label='y = -0.05x')
    ax2.plot([mn, mx], [0.05*mn, 0.05*mx], '--', color='blue', label='y = 0.05x')
    ax2.fill_between([mn, mx], [-0.05*mn, -0.05*mx], [0.05*mn, 0.05*mx], 
                      color='blue', alpha=0.2)
    ax2.set_xticks(unique_true_labels)
    ax2.set_xticklabels([f"{val:.2f}" for val in unique_true_labels], rotation=45)
    ax2.set_title("Error vs True Scattering Length (±5% Band)")
    ax2.set_xlabel("True Scattering Length (cm)")
    ax2.set_ylabel("Error (cm)")
    ax2.set_xlim(mn, mx)
    ax2.set_ylim(-0.1 * mx, 0.1 * mx)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3) Histogram of Errors (50 bins)
    ax3 = plt.subplot(2, 2, 3)
    bins = np.linspace(errors.min(), errors.max(), 50)
    ax3.hist(errors, bins=bins, color='green', alpha=0.7, edgecolor='black')
    ax3.set_title("Histogram of Errors")
    ax3.set_xlabel("Error (cm)")
    ax3.set_ylabel("Frequency")
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 4) Absolute Relative Error (%) by True with 68th Percentile
    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(true_vals, abs_rel_errors, s=10, color='purple', 
                label='Abs Rel Error (%)')
    perc68 = np.percentile(abs_rel_errors, 68)
    ax4.fill_between([mn, mx], 0, perc68, color='orange', alpha=0.3,
                     label=f'0–68th percentile ({perc68:.2f}%)')
    ax4.hlines(perc68, mn, mx, colors='red', linestyles='--',
               label=f'68th percentile = {perc68:.2f}%')
    ax4.set_xticks(unique_true_labels)
    ax4.set_xticklabels([f"{val:.2f}" for val in unique_true_labels], rotation=45)
    ax4.set_title("Absolute Relative Error (%) by True Scattering Length")
    ax4.set_xlabel("True Scattering Length (cm)")
    ax4.set_ylabel("Absolute Relative Error (%)")
    ax4.set_xlim(mn, mx)
    ax4.set_ylim(0, max(abs_rel_errors) * 1.1)
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """Plot training history (loss and learning rate).
    
    Args:
        history: Dictionary containing:
            - train_loss: List of training losses
            - val_loss: List of validation losses
            - lr: List of learning rates
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot losses
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot learning rate
    axes[1].plot(epochs, history['lr'], label='Learning Rate', 
                 marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()

