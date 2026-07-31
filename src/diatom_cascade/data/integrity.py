"""Reproducible dataset splits and image-integrity checks."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

from ..config import split_config


def resolve_model_dataset(data_root, model_type):
    mapping_file = Path(data_root) / "preprocessed" / "model_data_mapping.json"
    with mapping_file.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    if model_type not in mapping:
        raise ValueError(f"Model type '{model_type}' not found in {mapping_file}")
    return (mapping_file.parent / mapping[model_type]).resolve()


def split_manifest_dir(data_root, model_type):
    return Path(data_root) / "splits" / model_type.replace("-", "_")


def create_split_manifests(data_root, model_type, overwrite=False):
    """Create deterministic train/validation/test CSVs for one model."""
    from sklearn.model_selection import train_test_split

    output_dir = split_manifest_dir(data_root, model_type)
    paths = {name: output_dir / f"{name}.csv" for name in ("train", "validation", "test")}
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Split manifests already exist in {output_dir}; use --overwrite to replace them")

    df = pd.read_csv(resolve_model_dataset(data_root, model_type))
    target = split_config.STRATIFY_BY[model_type]
    if target not in df.columns:
        raise ValueError(f"Stratification column '{target}' missing for {model_type}")
    if df["filename"].duplicated().any():
        duplicates = df.loc[df["filename"].duplicated(), "filename"].tolist()[:10]
        raise ValueError(f"Duplicate filenames in {model_type}: {duplicates}")

    validate_images(df, Path(data_root) / "raw" / "images")

    train_val, test = train_test_split(
        df,
        test_size=split_config.TEST_RATIO,
        stratify=df[target],
        random_state=split_config.RANDOM_SEED,
    )
    train, validation = train_test_split(
        train_val,
        test_size=split_config.VAL_RATIO / (split_config.TRAIN_RATIO + split_config.VAL_RATIO),
        stratify=train_val[target],
        random_state=split_config.RANDOM_SEED,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame.sort_index().to_csv(paths[name], index=False)
    load_split_manifests(data_root, model_type)
    return paths


def load_split_manifests(data_root, model_type):
    """Load and validate the fixed manifests; never recreate them implicitly."""
    output_dir = split_manifest_dir(data_root, model_type)
    paths = {name: output_dir / f"{name}.csv" for name in ("train", "validation", "test")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing fixed split manifest(s): " + ", ".join(missing)
            + ". Run: python -m scripts.data.preprocessing.create_split_manifests"
        )

    splits = {name: pd.read_csv(path) for name, path in paths.items()}
    names = {name: set(frame["filename"].astype(str)) for name, frame in splits.items()}
    overlap = (names["train"] & names["validation"]) | (names["train"] & names["test"]) | (names["validation"] & names["test"])
    if overlap:
        raise ValueError(f"Filename leakage across {model_type} split manifests: {sorted(overlap)[:10]}")
    source = pd.read_csv(resolve_model_dataset(data_root, model_type))
    combined = pd.concat(splits.values(), ignore_index=True)
    try:
        pd.testing.assert_frame_equal(
            source.sort_values("filename").reset_index(drop=True),
            combined[source.columns].sort_values("filename").reset_index(drop=True),
            check_dtype=False,
        )
    except (AssertionError, KeyError) as exc:
        raise ValueError(f"Stale or altered split manifests for {model_type}; regenerate them explicitly") from exc
    return splits["train"], splits["validation"], splits["test"]


def validate_images(df, images_dir, report_path=None):
    """Fail if a referenced image is missing, unreadable, uniform, or byte-duplicated."""
    images_dir = Path(images_dir)
    failures = []
    content_hashes = defaultdict(list)
    for filename in sorted(set(df["filename"].astype(str))):
        image_path = images_dir / filename
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                extrema = image.convert("RGB").getextrema()
                if all(low == high for low, high in extrema):
                    raise ValueError(f"uniform image with channel extrema {extrema}")
            content_hashes[hashlib.sha256(image_path.read_bytes()).hexdigest()].append(filename)
        except (FileNotFoundError, OSError, ValueError) as exc:
            failures.append({"filename": filename, "error": f"{type(exc).__name__}: {exc}"})

    for digest, filenames in content_hashes.items():
        if len(filenames) > 1:
            joined = ", ".join(sorted(filenames))
            for filename in filenames:
                failures.append({
                    "filename": filename,
                    "error": f"DuplicateContent: SHA-256 {digest} shared by {joined}",
                })

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures, columns=["filename", "error"]).to_csv(report_path, index=False)
    if failures:
        location = f" See {report_path}." if report_path else ""
        preview = "; ".join(
            f"{failure['filename']}: {failure['error']}" for failure in failures[:3]
        )
        raise RuntimeError(
            f"Image preflight failed for {len(failures)} file(s): {preview}.{location}"
        )
    return True
