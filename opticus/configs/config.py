"""
Configuration class for OPTICUS
"""

import yaml
from pathlib import Path


class Config:
    """Configuration container"""
    
    def __init__(self, config_dict=None):
        """
        Initialize configuration
        
        Args:
            config_dict: Dictionary containing configuration
        """
        if config_dict is None:
            config_dict = {}
        
        self._config = config_dict
        
        # Set attributes for easy access
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def get(self, key, default=None):
        """Get configuration value"""
        return self._config.get(key, default)
    
    def __repr__(self):
        return f"Config({self._config})"
    
    def to_dict(self):
        """Convert to dictionary"""
        return self._config


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)

