"""Configuration loader and validator for YAML config files."""
import yaml
from pathlib import Path
from typing import Any, Optional


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Empty configuration file: {config_path}")

    return config


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge two configuration dictionaries.

    Override values take precedence over base values.
    Nested dictionaries are merged recursively.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_nested(config: dict, dotted_key: str, default: Optional[Any] = None) -> Any:
    """Retrieve a nested config value using dot notation.

    Example:
        get_nested(config, "model.xgboost.max_depth", 6)
    """
    keys = dotted_key.split(".")
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
