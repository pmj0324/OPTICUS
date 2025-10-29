"""
Dataset classes for OPTICUS
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class NoisyImageDataset(Dataset):
    """
    Dataset for noisy images with scattering length labels
    
    Args:
        images: (N, H, W) numpy array, raw pixel values (0~4095)
        labels: (N,) numpy array, raw labels
        lbl_min: Label minimum value for normalization
        lbl_max: Label maximum value for normalization
        transform: Optional transform to apply to images
    """
    
    def __init__(self, images, labels, lbl_min, lbl_max, transform=None):
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.lbl_min = lbl_min
        self.lbl_max = lbl_max
        self.transform = transform
        self.to_tensor = T.ToTensor()  # (H, W) → (1, H, W)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W)
        lbl = self.labels[idx]  # float

        # 1) Normalize pixel values [0, 4095] → [0, 1]
        img_norm = img / 4095.0

        # 2) Convert grayscale to tensor (1, H, W)
        img_tensor = self.to_tensor(img_norm)

        # 3) Apply optional transforms
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # 4) Normalize label to [-1, 1]
        lbl_norm = (lbl - self.lbl_min) / (self.lbl_max - self.lbl_min + 1e-8)  # [0, 1]
        lbl_scaled = lbl_norm * 2.0 - 1.0  # [-1, 1]
        lbl_tensor = torch.tensor(lbl_scaled, dtype=torch.float32)

        return img_tensor, lbl_tensor
    
    def denormalize_label(self, normalized_label):
        """
        Convert normalized label back to original scale
        
        Args:
            normalized_label: Label in [-1, 1] range
            
        Returns:
            Label in original scale
        """
        if isinstance(normalized_label, torch.Tensor):
            normalized_label = normalized_label.cpu().numpy()
        
        # [-1, 1] → [0, 1] → original scale
        lbl_01 = (normalized_label + 1.0) / 2.0
        original = lbl_01 * (self.lbl_max - self.lbl_min) + self.lbl_min
        return original

