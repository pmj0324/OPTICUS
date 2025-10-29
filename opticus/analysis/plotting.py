"""
Plotting functions for analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def setup_plot_style():
    """Setup consistent plotting style"""
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def plot_prediction_vs_true(true_vals, pred_vals, unit='m', save_path=None):
    """
    Plot predicted vs true values with ±5% band
    
    Args:
        true_vals: True values
        pred_vals: Predicted values
        unit: Unit for display ('m' or 'cm')
        save_path: Path to save figure (optional)
    """
    setup_plot_style()
    
    # Convert to cm if needed
    scale = 100 if unit == 'cm' else 1
    true_display = true_vals * scale
    pred_display = pred_vals * scale
    
    mn = min(true_display.min(), pred_display.min())
    mx = max(true_display.max(), pred_display.max())
    unique_labels = np.unique(true_display)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(true_display, pred_display, s=10, color='black', label='Predictions')
    
    # Reference lines
    ax.plot([mn, mx], [mn, mx], color='red', label='y = x')
    ax.plot([mn, mx], [0.95*mn, 0.95*mx], '--', color='blue', label='y = 0.95x')
    ax.plot([mn, mx], [1.05*mn, 1.05*mx], '--', color='blue', label='y = 1.05x')
    ax.fill_between([mn, mx], [0.95*mn, 0.95*mx], [1.05*mn, 1.05*mx], 
                     color='blue', alpha=0.2)
    
    # Formatting
    ax.set_xticks(unique_labels[::2])
    ax.set_xticklabels([f"{val:.0f}" if unit == 'cm' else f"{val:.2f}" 
                        for val in unique_labels[::2]], rotation=45)
    
    ax.set_title(f"Predicted vs True Scattering Length (±5% Band)")
    ax.set_xlabel(f"True Scattering Length ({unit})")
    ax.set_ylabel(f"Predicted Scattering Length ({unit})")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn * 0.9, mx * 1.1)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_distribution(true_vals, errors, unit='m', save_path=None):
    """
    Plot error distribution with ±5% band
    
    Args:
        true_vals: True values
        errors: Prediction errors
        unit: Unit for display ('m' or 'cm')
        save_path: Path to save figure (optional)
    """
    setup_plot_style()
    
    # Convert to cm if needed
    scale = 100 if unit == 'cm' else 1
    true_display = true_vals * scale
    errors_display = errors * scale
    
    mn = true_display.min()
    mx = true_display.max()
    unique_labels = np.unique(true_display)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(true_display, errors_display, s=10, color='black', 
               label='Prediction Errors')
    
    # Reference lines
    ax.plot([mn, mx], [0, 0], '-', color='red', label='No Error')
    ax.plot([mn, mx], [-0.05*mn, -0.05*mx], '--', color='blue', label='y = -0.05x')
    ax.plot([mn, mx], [0.05*mn, 0.05*mx], '--', color='blue', label='y = 0.05x')
    ax.fill_between([mn, mx], [-0.05*mn, -0.05*mx], [0.05*mn, 0.05*mx], 
                     color='blue', alpha=0.2)
    
    # Formatting
    ax.set_xticks(unique_labels[::2])
    ax.set_xticklabels([f"{val:.0f}" if unit == 'cm' else f"{val:.2f}" 
                        for val in unique_labels[::2]], rotation=45)
    
    ax.set_title(f"Error vs True Scattering Length (±5% Band)")
    ax.set_xlabel(f"True Scattering Length ({unit})")
    ax.set_ylabel(f"Error ({unit})")
    ax.set_xlim(mn, mx)
    ax.set_ylim(-0.1 * mx, 0.1 * mx)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_histogram(errors, unit='m', save_path=None):
    """
    Plot histogram of errors
    
    Args:
        errors: Prediction errors
        unit: Unit for display ('m' or 'cm')
        save_path: Path to save figure (optional)
    """
    setup_plot_style()
    
    # Convert to cm if needed
    scale = 100 if unit == 'cm' else 1
    errors_display = errors * scale
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    bins = np.linspace(errors_display.min(), errors_display.max(), 50)
    ax.hist(errors_display, bins=bins, color='green', alpha=0.7, edgecolor='black')
    
    ax.set_title("Histogram of Errors")
    ax.set_xlabel(f"Error ({unit})")
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_relative_error(true_vals, abs_rel_errors, unit='m', percentile=68, save_path=None):
    """
    Plot absolute relative error with percentile line
    
    Args:
        true_vals: True values
        abs_rel_errors: Absolute relative errors (%)
        unit: Unit for display ('m' or 'cm')
        percentile: Percentile to plot (default: 68)
        save_path: Path to save figure (optional)
    """
    setup_plot_style()
    
    # Convert to cm if needed
    scale = 100 if unit == 'cm' else 1
    true_display = true_vals * scale
    
    mn = true_display.min()
    mx = true_display.max()
    unique_labels = np.unique(true_display)
    
    # Calculate percentiles for each unique label
    perc_vals = []
    for lbl in unique_labels:
        mask = np.isclose(true_display, lbl, atol=1e-3)
        if np.any(mask):
            perc = np.percentile(abs_rel_errors[mask], percentile)
            perc_vals.append(perc)
        else:
            perc_vals.append(0)
    perc_vals = np.array(perc_vals)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(true_display, abs_rel_errors, s=10, color='purple', 
               label='Abs Rel Error (%)', alpha=0.45)
    
    # Percentile line
    ax.plot(unique_labels, perc_vals, color='red', lw=2, marker='o', 
            markersize=6, label=f'{percentile}th Percentile')
    
    # Formatting
    ax.set_xticks(unique_labels[::2])
    ax.set_xticklabels([f"{val:.0f}" if unit == 'cm' else f"{val:.2f}" 
                        for val in unique_labels[::2]], rotation=45)
    
    ax.set_title(f"Absolute Relative Error (%) by True Scattering Length")
    ax.set_xlabel(f"True Scattering Length ({unit})")
    ax.set_ylabel("Absolute Relative Error (%)")
    ax.set_xlim(mn, mx)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_metrics(metrics, unit='m', save_dir=None):
    """
    Plot all metrics in a single figure
    
    Args:
        metrics: Dictionary containing metrics from calculate_metrics
        unit: Unit for display ('m' or 'cm')
        save_dir: Directory to save individual plots (optional)
    """
    true_vals = metrics['true_vals']
    pred_vals = metrics['pred_vals']
    errors = metrics['errors']
    abs_rel_errors = metrics['abs_rel_errors']
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        plot_prediction_vs_true(true_vals, pred_vals, unit, 
                               f"{save_dir}/pred_vs_true.png")
        plot_error_distribution(true_vals, errors, unit, 
                               f"{save_dir}/error_distribution.png")
        plot_error_histogram(errors, unit, 
                            f"{save_dir}/error_histogram.png")
        plot_relative_error(true_vals, abs_rel_errors, unit, 68,
                           f"{save_dir}/relative_error.png")
    else:
        plot_prediction_vs_true(true_vals, pred_vals, unit)
        plot_error_distribution(true_vals, errors, unit)
        plot_error_histogram(errors, unit)
        plot_relative_error(true_vals, abs_rel_errors, unit, 68)


def save_all_plots(metrics, save_dir, unit='m'):
    """
    Save all plots to a directory
    
    Args:
        metrics: Dictionary containing metrics
        save_dir: Directory to save plots
        unit: Unit for display ('m' or 'cm')
    """
    plot_all_metrics(metrics, unit=unit, save_dir=save_dir)
    print(f"\nAll plots saved to {save_dir}/")

