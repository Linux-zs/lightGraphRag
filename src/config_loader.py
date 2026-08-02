"""Configuration loader with YAML, defaults, and env var overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.exceptions import ConfigError

# Load .env file if present
load_dotenv()

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
_LOCAL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "local.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides to config."""
    env_map = {
        "OLLAMA_HOST": ("ollama", "host"),
        "OLLAMA_MODEL": ("ollama", "model"),
        "LIGHTGRAPHRAG_DOCS_DIR": ("paths", "docs_dir"),
        "LIGHTGRAPHRAG_DATA_DIR": ("paths", "data_dir"),
    }
    for env_var, (section, key) in env_map.items():
        value = os.environ.get(env_var)
        if value:
            if section not in config:
                config[section] = {}
            config[section][key] = value
            logger.debug(f"Env override: {env_var} → config.{section}.{key}")
    return config


def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        logger.info(f"Loaded YAML config from {path}")
        return data
    except Exception as e:
        raise ConfigError(f"Failed to load YAML from {path}: {e}") from e


def load_config(config_path: Path | None = None) -> dict:
    """Load and merge configuration: YAML → local override → env overrides.

    Args:
        config_path: Optional custom config path. Defaults to config/default.yaml.
            When using the default config, config/local.yaml is loaded as a
            machine-local override if it exists. LIGHTGRAPHRAG_CONFIG_PATH can replace
            the main config path, and LIGHTGRAPHRAG_CONFIG_LOCAL_PATH can replace the
            local override path.

    Returns:
        Merged configuration dictionary.

    Raises:
        ConfigError: If config file cannot be loaded.
    """
    env_config_path = os.environ.get("LIGHTGRAPHRAG_CONFIG_PATH")
    path = Path(env_config_path) if env_config_path else (config_path or _DEFAULT_CONFIG_PATH)
    config = load_yaml(path)
    default_path = _DEFAULT_CONFIG_PATH.resolve()
    if path.resolve() == default_path:
        local_path = Path(
            os.environ.get("LIGHTGRAPHRAG_CONFIG_LOCAL_PATH", _LOCAL_CONFIG_PATH)
        )
        if local_path.exists():
            config = _deep_merge(config, load_yaml(local_path))
    config = _apply_env_overrides(config)
    # Ensure data_dir derived paths
    data_dir = config.get("paths", {}).get("data_dir", "./data")
    paths = config.setdefault("paths", {})
    if "log_dir" not in paths:
        paths["log_dir"] = str(Path(data_dir) / "logs")
    logger.info("Configuration loaded and merged successfully")
    return config


# Global config singleton
_config: dict | None = None


def get_config() -> dict:
    """Get the global configuration (lazy loaded)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the global config for testing or reload."""
    global _config
    _config = None
