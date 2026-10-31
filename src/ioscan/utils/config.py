"""YAML config loading."""
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    """Repo root, assuming src/ioscan/utils/config.py layout."""
    return Path(__file__).resolve().parents[3]
