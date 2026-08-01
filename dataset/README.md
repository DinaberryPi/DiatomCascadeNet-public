# Dataset layout

The repository includes this empty directory structure but does not distribute
images, annotations, split manifests, or derived data. The original study's
third-party source images are withheld because redistribution rights are not
available. The repository's MIT licence applies to code, not to those data.

```text
dataset/
  raw/
    images/
    metadata.xlsx
    labels.csv
  exclusions/
    invalid_images.csv
  cleaned/
  preprocessed/
  splits/
```

For the standard workflow, provide:

- `raw/images/`: appropriately licensed image files.
- `raw/metadata.xlsx`: one row per image with `filename`, `class`, `order`,
  `family`, `genus`, and `species` columns. The `filename` values must exactly
  match files in `raw/images/`.
- `exclusions/invalid_images.csv` (optional): a `filename` column listing images
  to exclude. A header-only file represents no exclusions.

For archival compatibility, the converter also accepts the original Chinese
columns `ppt序号`, `纲`, `目`, `科`, `属名`, and `种名`; filenames are then generated
as `slide_0001.png`, `slide_0002.png`, and so on.

`scripts.data.labeling.convert_metadata` creates `raw/labels.csv`. The cleaning,
filtering, taxonomy, and split scripts create the remaining artifacts. All data
files are ignored by Git.
