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

- `raw/images/`: appropriately licensed, non-uniform images prepared at exactly
  `320 x 320` pixels. PNG and RGB are recommended; any Pillow-readable format
  referenced by the workbook is accepted.
- `raw/metadata.xlsx`: one row per image with `filename`, `class`, `order`,
  `family`, `genus`, and `species` columns. The `filename` values must exactly
  match files in `raw/images/`.
- `exclusions/invalid_images.csv` (optional): a `filename` column listing images
  to exclude. A header-only file represents no exclusions.

For archival compatibility, the converter also accepts the original Chinese
columns `ppt序号`, `纲`, `目`, `科`, `属名`, and `种名`; filenames are then generated
as `slide_0001.png`, `slide_0002.png`, and so on. New datasets should use the
English schema above.

## Image preparation

The audited study inputs contained 4,869 RGB PNG files, all exactly `320 x 320`
pixels. Keep `IMAGE_SIZE = 320` to reproduce or directly compare with the
study. Users exploring their own datasets may select another square input size
in `src/diatom_cascade/config/data_config.py` and prepare all final images at
that size. Results from another size are a new experiment, not a strict
reproduction of the study.

- Use one centered diatom specimen per image.
- Crop the specimen consistently, preserve its aspect ratio, and pad to a
  square canvas. Do not stretch a non-square crop.
- Prefer source crops of at least `320 x 320`; upscaling smaller images cannot
  restore lost detail.
- Keep magnification, illumination, background, color handling, and scale-bar
  treatment as consistent as possible. Remove labels, page text, and borders
  that could reveal the class.
- Do not place multiple crops or views from the same physical specimen in
  different train, validation, or test splits.

The original helper `scripts/data/cleaning/prep_images.py` records the study's
ROI crop, aspect-preserving resize, and white-padding procedure. Public users
may prepare images with another tool, but the resulting files must satisfy the
same input contract before split creation.

`scripts.data.labeling.convert_metadata` creates `raw/labels.csv`. The cleaning,
filtering, taxonomy, and split scripts create the remaining artifacts. All data
files are ignored by Git.
