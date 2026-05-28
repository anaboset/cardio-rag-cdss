import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Any

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _interpolate_env_vars(config: Any):
    """
    Recursively look for strings like '${VAR_NAME}' in the config 
    and replace them with actual environment variables.
    """
    
    if isinstance(config, dict):
        for k, v in config.items():
            config[k] = _interpolate_env_vars(v)
    elif isinstance(config, list):
        return [_interpolate_env_vars(i) for i in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.getenv(env_var, config) 
    return config


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load and interpolate the main configuration.
    """
    full_path = PROJECT_ROOT / config_path
    config = load_yaml(full_path)

    interpolated = _interpolate_env_vars(config)
    
    if not isinstance(interpolated, dict):
        raise TypeError(f"Config at {full_path} must be a YAML dictionary.")
    
    return interpolated


def load_prompt(template_key: str) -> str:
    """
    Load a prompt template content based on a key in prompts.yaml.
    """
    prompts_map = load_yaml(PROJECT_ROOT / "config/prompts.yaml")
    
    if template_key not in prompts_map:
        raise KeyError(f"Prompt key '{template_key}' not found in prompts.yaml")
    
    prompt_path = PROJECT_ROOT / prompts_map[template_key]
    with open(prompt_path, "r") as f:
        return f.read()

CONFIG = load_config()