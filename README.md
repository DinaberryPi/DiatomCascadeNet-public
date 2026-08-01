# DiatomCascadeNet

Research code for *Hierarchical Deep Learning for Diatom Image Classification: A Multi-Level Taxonomic Approach*.

- Paper: [arXiv:2512.06613](https://arxiv.org/abs/2512.06613)
- Research site: [DiatomCascadeNet Research](https://diatom-cascade-net.yueying-dina-ke.chatgpt.site)
- Output-free Colab template: [`notebooks/DiatomScanNet_2026_reproducible.ipynb`](notebooks/DiatomScanNet_2026_reproducible.ipynb)

## Result boundary

Private archived evaluation artifacts report 69.4% species accuracy for both H-COFGS and F-S on the same 219-image test manifest. A data-only recomputation reports a reduction in mean taxonomic error distance from 1.955 (F-S) to 1.209 (H-COFGS), approximately 38.2%. Prediction files, checkpoints, and source images are not distributed here.

These are archived results, not outputs of the refactored 2026 pipeline. The canonical pipeline must be rerun before its outputs replace the archived paper values. Beam-search results remain excluded until regenerated and independently tested.

## Architecture contract

- `src/diatom_cascade/models.py` is the only model-definition module.
- Training, evaluation, and prediction import the same canonical classes.
- Every hierarchical stage creates new classifier heads.
- H-CO, H-COF, H-COFG, and H-COFGS strictly transfer only `backbone.*` from the immediately preceding checkpoint.
- The backbone remains trainable after transfer.
- Fixed train, validation, and test manifests are created once and reused by all execution paths.
- Image preflight fails on missing, unreadable, uniform, or byte-duplicated files.

Pre-refactor checkpoints with stage-specific module names such as `class_backbone.*` are intentionally rejected by the canonical schema. Use a fresh output directory for the rerun.

## Repository map

```text
src/diatom_cascade/   Importable models, config, checkpoint, data, loss, and decoding code
scripts/data/         Cleaning, taxonomy extraction, filtering, and split creation
scripts/train/        Seven model-training entry points
scripts/evaluate/     Frozen-test evaluation entry points
scripts/predict/      Single-model inference entry points
scripts/analysis/     Tables, figures, and training-curve workflows
scripts/experiments/  Error-propagation analyses
notebooks/            Output-free Colab execution template
tests/                Fail-closed, architecture, path, and data-integrity checks
site/                 Bilingual public research website
```

The empty `dataset/` directory structure is included so the same paths are used in every environment. Its images, annotations, generated tables, and split manifests remain ignored by Git. Private `outputs/`, `docs/`, and `report/` trees are also ignored.

## Reproducible run

Run commands from the repository root. The Google Colab notebook follows the original 2025 setup: clone the repository, install its requirements, mount Google Drive, and copy the private inputs into the ignored `dataset/` directory.

```bash
python -m pip install -e .
python -m unittest discover -s tests -p "test_*.py" -v
python -m scripts.data.cleaning.clean_data
python -m scripts.data.preprocessing.build_taxonomy_tree
python -m scripts.data.preprocessing.create_filtered_datasets
python -m scripts.data.preprocessing.create_split_manifests --overwrite
```

The Colab notebook then runs the progressive sequence F-C to H-CO to H-COF to H-COFG to H-COFGS, trains the independent F-G and F-S baselines, and evaluates all seven frozen checkpoints.

The template expects the same Drive folder used by the 2025 notebook, with two additional private files that are no longer stored in the repository:

```text
MyDrive/DiatomScanNet/
  images/
  metadata.xlsx
  invalid_images.csv
  runs/<RUN_ID>/
```

The notebook rebuilds `labels.csv`, the deterministic taxonomy tree, filtered tables, and fixed split manifests inside Colab. Run-specific checkpoints, logs, predictions, evaluations, and hashes are written to Drive under `runs/<RUN_ID>/`.

## Data policy

- The complete dataset, labels, split manifests, checkpoints, logs, predictions, and internal reports are private research artifacts.
- The public notebook contains no outputs, attachments, embedded images, or user-specific private paths. It includes only a generic, editable Google Drive layout.
- The executed 2025 notebook remains a private local archive.
- The website has no image uploader or live classifier.

## Website development

```bash
cd site
npm install
npm run dev
```

Use `npm test` for the production build and publication-boundary checks.

## License

The source code is released under the [MIT License](LICENSE). This license does not grant rights to the private dataset, source images, metadata, experiment artifacts, or model checkpoints.
