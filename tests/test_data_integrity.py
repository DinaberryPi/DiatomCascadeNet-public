import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from diatom_cascade.config.data_config import IMAGE_SIZE
from diatom_cascade.data.integrity import validate_images


class ValidateImagesTests(unittest.TestCase):
    def test_rejects_wrong_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE - 1), "white")
            image.putpixel((0, 0), (0, 0, 0))
            image.save(Path(directory) / "wrong-size.png")
            with self.assertRaisesRegex(RuntimeError, f"expected {IMAGE_SIZE}x{IMAGE_SIZE}"):
                validate_images(
                    pd.DataFrame({"filename": ["wrong-size.png"]}), directory
                )

    def test_rejects_uniform_image(self):
        with tempfile.TemporaryDirectory() as directory:
            Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white").save(
                Path(directory) / "blank.png"
            )
            with self.assertRaisesRegex(RuntimeError, "uniform image"):
                validate_images(pd.DataFrame({"filename": ["blank.png"]}), directory)

    def test_rejects_duplicate_content(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
            image.putpixel((0, 0), (0, 0, 0))
            image.save(Path(directory) / "a.png")
            image.save(Path(directory) / "b.png")
            with self.assertRaisesRegex(RuntimeError, "DuplicateContent"):
                validate_images(pd.DataFrame({"filename": ["a.png", "b.png"]}), directory)

    def test_accepts_distinct_nonuniform_images(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
            first.putpixel((0, 0), (0, 0, 0))
            first.save(Path(directory) / "a.png")
            second = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
            second.putpixel((1, 1), (0, 0, 0))
            second.save(Path(directory) / "b.png")
            self.assertTrue(validate_images(
                pd.DataFrame({"filename": ["a.png", "b.png"]}), directory
            ))


if __name__ == "__main__":
    unittest.main()
