"""
Data loading utility functions
"""

import random
import numpy as np
import h5py
from torch.utils.data import DataLoader, Subset

from .dataset import NoisyImageDataset


def load_h5_data(hdf5_path):
    """
    Load images and labels from HDF5 file
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        tuple: (images, labels, lbl_min, lbl_max)
    """
    with h5py.File(hdf5_path, 'r') as f:
        images = f['images'][:]  # (N, H, W)
        labels = f['labels'][:]  # (N,)
    
    lbl_min = labels.min()
    lbl_max = labels.max()
    
    return images, labels, lbl_min, lbl_max


def create_dataloaders(
    images,
    labels,
    lbl_min,
    lbl_max,
    batch_size=32,
    train_split=0.6,
    val_split=0.2,
    test_split=0.2,
    num_workers=4,
    pin_memory=True,
    seed=None
):
    """
    Create train/val/test dataloaders with stratified splitting
    
    Args:
        images: Image array
        labels: Label array
        lbl_min: Minimum label value
        lbl_max: Maximum label value
        batch_size: Batch size for dataloaders
        train_split: Training split ratio (default: 0.6)
        val_split: Validation split ratio (default: 0.2)
        test_split: Test split ratio (default: 0.2)
        num_workers: Number of workers for dataloader
        pin_memory: Whether to pin memory
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create dataset
    dataset = NoisyImageDataset(images, labels, lbl_min, lbl_max)
    
    # Get unique labels for stratified splitting
    unique_labels = np.unique(labels)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Stratified split: split each label class separately
    for lbl in unique_labels:
        indices = np.where(labels == lbl)[0].tolist()
        random.shuffle(indices)
        
        n = len(indices)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        
        test_indices.extend(indices[:n_test])
        val_indices.extend(indices[n_test:n_test + n_val])
        train_indices.extend(indices[n_test + n_val:])
    
    # Create dataloaders
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Dataset split - Train: {len(train_indices)}, "
          f"Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return train_loader, val_loader, test_loader, dataset

