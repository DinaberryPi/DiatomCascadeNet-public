import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from diatom_cascade.data.integrity import validate_images


class ValidateImagesTests(unittest.TestCase):
    def test_rejects_uniform_image(self):
        with tempfile.TemporaryDirectory() as directory:
            Image.new("RGB", (8, 8), "white").save(Path(directory) / "blank.png")
            with self.assertRaisesRegex(RuntimeError, "uniform image"):
                validate_images(pd.DataFrame({"filename": ["blank.png"]}), directory)

    def test_rejects_duplicate_content(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new("RGB", (8, 8), "white")
            image.putpixel((0, 0), (0, 0, 0))
            image.save(Path(directory) / "a.png")
            image.save(Path(directory) / "b.png")
            with self.assertRaisesRegex(RuntimeError, "DuplicateContent"):
                validate_images(pd.DataFrame({"filename": ["a.png", "b.png"]}), directory)

    def test_accepts_distinct_nonuniform_images(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Image.new("RGB", (8, 8), "white")
            first.putpixel((0, 0), (0, 0, 0))
            first.save(Path(directory) / "a.png")
            second = Image.new("RGB", (8, 8), "white")
            second.putpixel((1, 1), (0, 0, 0))
            second.save(Path(directory) / "b.png")
            self.assertTrue(validate_images(
                pd.DataFrame({"filename": ["a.png", "b.png"]}), directory
            ))


if __name__ == "__main__":
    unittest.main()
