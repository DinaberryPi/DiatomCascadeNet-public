import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.data.cleaning import clean_data
from scripts.evaluate import run_all_evaluations


class FailClosedTests(unittest.TestCase):
    def test_cleaning_accepts_empty_exclusion_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_labels = root / "labels.csv"
            manifest = root / "invalid_images.csv"
            clean_labels = root / "cleaned" / "labels_clean.csv"
            pd.DataFrame(
                [{
                    "filename": "sample.png",
                    "class": "Class A",
                    "order": "Order A",
                    "family": "Family A",
                    "genus": "Genus A",
                    "species": "Species A",
                }]
            ).to_csv(raw_labels, index=False)
            pd.DataFrame(columns=["filename"]).to_csv(manifest, index=False)

            with mock.patch.object(clean_data, "RAW_LABELS", raw_labels), mock.patch.object(
                clean_data, "INVALID_IMAGES", manifest
            ), mock.patch.object(clean_data, "CLEAN_LABELS", clean_labels), mock.patch.object(
                clean_data, "DATA_ROOT", root
            ):
                clean_data.main()

            self.assertTrue(clean_labels.is_file())

    def test_cleaning_requires_exclusion_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_labels = root / "labels.csv"
            pd.DataFrame([{"filename": "sample.png"}]).to_csv(raw_labels, index=False)

            with mock.patch.object(clean_data, "RAW_LABELS", raw_labels), mock.patch.object(
                clean_data, "INVALID_IMAGES", root / "missing.csv"
            ):
                with self.assertRaises(FileNotFoundError):
                    clean_data.main()

    def test_cleaning_rejects_malformed_exclusion_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_labels = root / "labels.csv"
            manifest = root / "invalid_images.csv"
            pd.DataFrame([{"filename": "sample.png"}]).to_csv(raw_labels, index=False)
            pd.DataFrame([{"wrong_column": "sample.png"}]).to_csv(manifest, index=False)

            with mock.patch.object(clean_data, "RAW_LABELS", raw_labels), mock.patch.object(
                clean_data, "INVALID_IMAGES", manifest
            ):
                with self.assertRaisesRegex(ValueError, "Invalid exclusion manifest"):
                    clean_data.main()

    def test_full_evaluation_requires_every_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            run_all_evaluations, "get_output_dir", return_value=Path(temp_dir)
        ), mock.patch.object(sys, "argv", ["run_all_evaluations.py"]):
            with self.assertRaises(FileNotFoundError):
                run_all_evaluations.main()


if __name__ == "__main__":
    unittest.main()
