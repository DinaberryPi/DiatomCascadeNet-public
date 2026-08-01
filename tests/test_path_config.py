import os
import unittest
from pathlib import Path
from unittest.mock import patch


from diatom_cascade.config.path_config import get_data_root, get_output_dir


class OutputPathTests(unittest.TestCase):
    def test_external_data_directory(self):
        with patch.dict(os.environ, {"DIATOM_DATA_ROOT": "private/dataset"}):
            self.assertEqual(get_data_root(), Path("private/dataset"))

    def test_default_output_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_output_dir(), Path("outputs"))

    def test_run_specific_output_directory(self):
        with patch.dict(os.environ, {"DIATOM_OUTPUT_DIR": "outputs/runs/clean_2026"}):
            self.assertEqual(get_output_dir(), Path("outputs/runs/clean_2026"))


if __name__ == "__main__":
    unittest.main()
