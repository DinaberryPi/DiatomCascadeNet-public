# Dataset layout

The repository includes this empty directory structure but does not distribute
images, annotations, split manifests, or derived data.

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

- `raw/images/`: image files named `slide_0001.png`, `slide_0002.png`, and so on.
- `raw/metadata.xlsx`: annotations containing `ppt序号`, `纲`, `目`, `科`, `属名`, and `种名` columns.
- `exclusions/invalid_images.csv`: one `filename` column listing images to exclude. A header-only file is valid when there are no exclusions.

`scripts.data.labeling.convert_metadata` creates `raw/labels.csv`. The cleaning,
filtering, taxonomy, and split scripts create the remaining artifacts. All data
files are ignored by Git.
