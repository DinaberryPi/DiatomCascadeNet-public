"""Dependency-free artifact path configuration."""

import os
from pathlib import Path


def get_project_root():
    """Return the repository root and fail if a workflow runs from elsewhere."""
    root = Path(os.environ.get("DIATOM_PROJECT_ROOT", Path.cwd())).resolve()
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(
            "Run workflows from the repository root or set DIATOM_PROJECT_ROOT"
        )
    return root


def get_output_dir(project_root=None):
    value = os.environ.get("DIATOM_OUTPUT_DIR", "outputs").strip()
    if not value:
        raise ValueError("DIATOM_OUTPUT_DIR must not be empty")
    path = Path(value)
    if project_root is not None and not path.is_absolute():
        path = Path(project_root) / path
    return path


def get_data_root(project_root=None):
    """Return the private dataset root without requiring it inside the repository."""
    value = os.environ.get("DIATOM_DATA_ROOT", "dataset").strip()
    if not value:
        raise ValueError("DIATOM_DATA_ROOT must not be empty")
    path = Path(value)
    if project_root is not None and not path.is_absolute():
        path = Path(project_root) / path
    return path
