"""Configuration management for butterfly classification project."""

import os
from pathlib import Path
from typing import Dict, Any
import yaml


class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.config_path = config_path or self.project_root / "config" / "model_config.yaml"
        self._config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'image_size': [224, 224],
                'batch_size': 32,
                'validation_split': 0.2,
                'num_classes': 75
            },
            'training': {
                'epochs': 50,
                'learning_rate': 1e-4,
                'early_stopping_patience': 5,
                'reduce_lr_patience': 3,
                'reduce_lr_factor': 0.5
            },
            'models': {
                'fnn': {
                    'hidden_layers': [1024, 512, 256, 128],
                    'dropout_rates': [0.5, 0.4, 0.3, 0.2]
                },
                'cnn': {
                    'filters': [32, 64, 128],
                    'kernel_size': 3,
                    'pool_size': 2,
                    'dense_units': 512,
                    'dropout_rate': 0.5
                },
                'vgg16': {
                    'dense_units': 512,
                    'dropout_rate': 0.3,
                    'batch_norm': True
                }
            },
            'paths': {
                'data_dir': 'data',
                'models_dir': 'models',
                'logs_dir': 'logs'
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save_config(self, path: str = None):
        """Save configuration to file."""
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)


# Global config instance
config = Config()