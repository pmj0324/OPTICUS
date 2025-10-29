"""
Analysis and visualization tools for OPTICUS
"""

from .plotting import (
    plot_prediction_vs_true,
    plot_error_distribution,
    plot_error_histogram,
    plot_relative_error,
    plot_all_metrics,
    save_all_plots
)

__all__ = [
    'plot_prediction_vs_true',
    'plot_error_distribution',
    'plot_error_histogram',
    'plot_relative_error',
    'plot_all_metrics',
    'save_all_plots'
]

