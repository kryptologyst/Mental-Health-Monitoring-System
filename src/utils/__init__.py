"""
Core utilities for mental health monitoring system.

This module provides essential utilities including device detection,
deterministic seeding, and configuration management.
"""

import os
import random
import logging
from typing import Any, Dict, Optional, Union
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device(device_name: Optional[str] = None) -> torch.device:
    """Get the best available device for computation.
    
    Args:
        device_name: Specific device name (cuda, mps, cpu).
        
    Returns:
        PyTorch device object.
    """
    if device_name:
        return torch.device(device_name)
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    return OmegaConf.load(config_path)


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "mental_health_monitoring.log")),
            logging.StreamHandler()
        ]
    )


def validate_config(config: DictConfig) -> None:
    """Validate configuration parameters.
    
    Args:
        config: Configuration object to validate.
        
    Raises:
        ValueError: If configuration is invalid.
    """
    required_keys = ["model", "training", "data", "evaluation"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate model configuration
    if config.model.num_labels <= 0:
        raise ValueError("Number of labels must be positive")
    
    if config.model.max_length <= 0:
        raise ValueError("Max length must be positive")
    
    # Validate training configuration
    if config.training.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    if config.training.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    # Validate data splits
    total_split = config.data.train_split + config.data.val_split + config.data.test_split
    if not np.isclose(total_split, 1.0, atol=1e-6):
        raise ValueError(f"Data splits must sum to 1.0, got {total_split}")


class ConfigManager:
    """Configuration manager for the mental health monitoring system."""
    
    def __init__(self, config_path: str):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = load_config(config_path)
        validate_config(self.config)
        
        # Set up logging
        setup_logging(
            log_level=self.config.logging.level,
            log_dir=self.config.logging.log_dir
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Configuration loaded from {config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        return OmegaConf.select(self.config, key, default=default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates.
        """
        self.config = OmegaConf.merge(self.config, updates)
        self.logger.info(f"Configuration updated with: {list(updates.keys())}")


def format_confidence(confidence: float, decimals: int = 2) -> str:
    """Format confidence score for display.
    
    Args:
        confidence: Confidence score between 0 and 1.
        decimals: Number of decimal places.
        
    Returns:
        Formatted confidence string.
    """
    return f"{confidence:.{decimals}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value.
        denominator: Denominator value.
        default: Default value if division by zero.
        
    Returns:
        Division result or default value.
    """
    return numerator / denominator if denominator != 0 else default
