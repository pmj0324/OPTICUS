"""Dataset and DataLoader utilities for OPTICUS.
OPTICUS용 데이터셋 및 데이터로더 유틸리티.
"""
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class NoisyImageDataset(Dataset):
    """Dataset for noisy 500×500 images with labels.
    레이블이 있는 노이즈 포함 500×500 이미지 데이터셋.
    
    Args / 인자:
        images: (N, 500, 500) numpy array, raw pixel values (0~4095)
               (N, 500, 500) numpy 배열, 원시 픽셀 값 (0~4095)
        labels: (N,) numpy array, raw labels
               (N,) numpy 배열, 원시 레이블
        lbl_min: Label minimum for normalization
                정규화를 위한 레이블 최소값
        lbl_max: Label maximum for normalization
                정규화를 위한 레이블 최대값
    """
    
    def __init__(self, images, labels, lbl_min, lbl_max):
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.lbl_min = lbl_min
        self.lbl_max = lbl_max
        self.to_tensor = T.ToTensor()  # (H, W) → (1, H, W)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]   # (500, 500)
        lbl = self.labels[idx]   # float

        # 1) Normalize [0,4095] → [0,1]
        img_norm = img / 4095.0

        # 2) Grayscale → Tensor (1, 500, 500)
        img_tensor = self.to_tensor(img_norm)

        # 3) Label Min–Max normalization → [-1,1]
        lbl_norm = (lbl - self.lbl_min) / (self.lbl_max - self.lbl_min + 1e-8)  # [0,1]
        lbl_scaled = lbl_norm * 2.0 - 1.0                                       # [-1,1]
        lbl_tensor = torch.tensor(lbl_scaled, dtype=torch.float32)

        return img_tensor, lbl_tensor


def load_dataset(path, lbl_min, lbl_max):
    """Load dataset from HDF5 file.
    
    Args:
        path: Path to HDF5 file
        lbl_min: Label minimum for normalization
        lbl_max: Label maximum for normalization
        
    Returns:
        NoisyImageDataset instance
    """
    with h5py.File(path, 'r') as f:
        images = f['images'][:]
        labels = f['labels'][:]
    return NoisyImageDataset(images, labels, lbl_min, lbl_max)


def compute_label_stats(train_path, val_path, test_path):
    """Compute global min/max statistics from train, val, and test labels.
    
    Args:
        train_path: Path to training HDF5 file
        val_path: Path to validation HDF5 file
        test_path: Path to test HDF5 file
        
    Returns:
        Tuple of (lbl_min, lbl_max)
    """
    all_labels = []
    for path in [train_path, val_path, test_path]:
        with h5py.File(path, 'r') as f:
            all_labels.append(f['labels'][:])
    all_labels = np.concatenate(all_labels)
    lbl_min = all_labels.min()
    lbl_max = all_labels.max()
    return lbl_min, lbl_max


def create_dataloaders(train_path, val_path, test_path, batch_size=32, 
                       num_workers=4, pin_memory=True, seed=None):
    """Create train, validation, and test dataloaders.
    
    Args:
        train_path: Path to training HDF5 file
        val_path: Path to validation HDF5 file
        test_path: Path to test HDF5 file
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        pin_memory: Whether to use pinned memory
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, lbl_min, lbl_max)
    """
    # Compute label statistics
    lbl_min, lbl_max = compute_label_stats(train_path, val_path, test_path)
    
    # Load datasets
    train_dataset = load_dataset(train_path, lbl_min, lbl_max)
    val_dataset = load_dataset(val_path, lbl_min, lbl_max)
    test_dataset = load_dataset(test_path, lbl_min, lbl_max)
    
    # Worker seed function
    def seed_worker(worker_id):
        if seed is not None:
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            import random
            random.seed(worker_seed)
    
    # Generator for reproducibility
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader, lbl_min, lbl_max

