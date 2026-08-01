import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from openpyxl import Workbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MetadataConversionTests(unittest.TestCase):
    def test_accepts_public_english_metadata_format(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            raw_dir = data_root / "raw"
            raw_dir.mkdir()

            workbook = Workbook()
            worksheet = workbook.active
            worksheet.append([
                "filename", "class", "order", "family", "genus", "species"
            ])
            worksheet.append([
                "sample.png", "C", "O", "F", "G", "S"
            ])
            workbook.save(raw_dir / "metadata.xlsx")

            environment = os.environ.copy()
            environment["DIATOM_PROJECT_ROOT"] = str(PROJECT_ROOT)
            environment["DIATOM_DATA_ROOT"] = str(data_root)
            subprocess.run(
                [sys.executable, "-m", "scripts.data.labeling.convert_metadata"],
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
            )

            labels = pd.read_csv(raw_dir / "labels.csv")
            self.assertEqual(labels.to_dict("records"), [{
                "filename": "sample.png",
                "class": "C",
                "order": "O",
                "family": "F",
                "genus": "G",
                "species": "S",
            }])


if __name__ == "__main__":
    unittest.main()
