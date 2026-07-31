"""Strict helpers for trusted DiatomCascadeNet checkpoints."""

from collections.abc import Mapping

import torch


CHECKPOINT_SCHEMA_VERSION = 3
BACKBONE_PREFIX = "backbone."


def load_trusted_checkpoint(checkpoint_path, map_location="cpu"):
    """Load a project checkpoint without arbitrary object deserialization."""
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    return checkpoint


def extract_model_state_dict(checkpoint):
    """Return the required model state dictionary."""
    try:
        model_state_dict = checkpoint["model_state_dict"]
    except KeyError as error:
        raise KeyError("Checkpoint is missing required 'model_state_dict'") from error
    if not isinstance(model_state_dict, Mapping):
        raise TypeError("'model_state_dict' must be a mapping")
    return model_state_dict


def validate_checkpoint_schema(checkpoint):
    """Reject checkpoints produced by a different architecture contract."""
    version = checkpoint.get("checkpoint_schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Checkpoint schema mismatch: "
            f"expected {CHECKPOINT_SCHEMA_VERSION}, found {version!r}. "
            "Use checkpoints created by the canonical 2026 pipeline."
        )
    return checkpoint


def load_backbone_from_checkpoint(checkpoint_path, target_module, map_location="cpu"):
    """Strictly transfer the canonical backbone from the previous stage."""
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=map_location)
    validate_checkpoint_schema(checkpoint)
    model_state_dict = extract_model_state_dict(checkpoint)
    backbone_state_dict = {
        key[len(BACKBONE_PREFIX) :]: value
        for key, value in model_state_dict.items()
        if isinstance(key, str) and key.startswith(BACKBONE_PREFIX)
    }
    if not backbone_state_dict:
        raise KeyError(
            f"No model_state_dict keys found under required prefix '{BACKBONE_PREFIX}'"
        )
    target_module.load_state_dict(backbone_state_dict, strict=True)
    return checkpoint
