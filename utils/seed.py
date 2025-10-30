"""Seed utilities for reproducibility.
재현성을 위한 시드 유틸리티.
"""
import random
import numpy as np
import torch


def set_seed(seed):
    """Set random seed for reproducibility.
    재현성을 위한 랜덤 시드 설정.
    
    Args / 인자:
        seed: Random seed value
             랜덤 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

