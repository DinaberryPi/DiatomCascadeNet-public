import json
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRECTORIES = {
    "dataset",
    "docs",
    "local_audit",
    "outputs",
    "report",
    "site-review",
}
PRIVATE_SUFFIXES = {
    ".bmp",
    ".ckpt",
    ".csv",
    ".gif",
    ".h5",
    ".hdf5",
    ".joblib",
    ".jpeg",
    ".jpg",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".tif",
    ".tiff",
    ".tsv",
    ".webp",
    ".xlsx",
}


def git_paths(*arguments):
    output = subprocess.check_output(
        ["git", *arguments, "-z"],
        cwd=PROJECT_ROOT,
    )
    return {
        Path(item.decode("utf-8"))
        for item in output.split(b"\0")
        if item
    }


class PublicBoundaryTests(unittest.TestCase):
    def test_git_candidates_exclude_private_artifacts(self):
        candidates = git_paths("ls-files") | git_paths(
            "ls-files", "--others", "--exclude-standard"
        )
        violations = []
        for path in candidates:
            if path.parts and path.parts[0] in PRIVATE_DIRECTORIES:
                violations.append(str(path))
            elif path.suffix.lower() in PRIVATE_SUFFIXES:
                violations.append(str(path))
            elif path.name == "DiatomScanNet_2025.ipynb":
                violations.append(str(path))
        self.assertEqual(sorted(violations), [])

    def test_public_notebook_has_no_outputs_or_attachments(self):
        path = PROJECT_ROOT / "notebooks" / "DiatomScanNet_2026_reproducible.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            with self.subTest(cell=index):
                self.assertFalse(cell.get("attachments"))
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs"), [])

    def test_public_text_has_no_private_absolute_path(self):
        forbidden = (
            "C:" + "\\Users\\" + "dinah",
            "/content/" + "drive",
            "My" + "Drive",
        )
        suffixes = {".css", ".ipynb", ".js", ".json", ".md", ".mjs", ".py", ".toml", ".ts", ".tsx"}
        violations = []
        for path in PROJECT_ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if any(part in {".git", "dist", "node_modules"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(value in text for value in forbidden):
                violations.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
