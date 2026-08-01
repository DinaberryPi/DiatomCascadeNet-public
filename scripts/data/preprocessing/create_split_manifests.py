#!/usr/bin/env python3
"""Create fixed train/validation/test manifests for every model."""

import argparse
import sys
from pathlib import Path


from diatom_cascade.config import split_config
from diatom_cascade.config.path_config import get_data_root
from diatom_cascade.data.integrity import create_split_manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for model_type in split_config.STRATIFY_BY:
        paths = create_split_manifests(get_data_root(), model_type, overwrite=args.overwrite)
        print(f"{model_type}: " + ", ".join(str(path) for path in paths.values()))


if __name__ == "__main__":
    main()
