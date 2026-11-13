# Changelog

## Unreleased: repository restructure

- Weights and results are organised by version: `weights/v1`, `weights/v2`, `results/v1`, `results/v2`. The board
  comparison is in `results/jetson_board_100`, which now also holds its source table (`all_models.csv`) and bootstrap
  intervals.
- New docs: `docs/architecture.md`, `docs/results.md` and `docs/installation.md`. The README is shorter, and the v2
  board table now includes GIT-base-COCO.
- `pyproject.toml` makes the package installable, with `eval` and `dev` extras.
- `caption.py` works with weights that have no full-encoder connector (v2). A missing weight file or connector now
  gives a clear error.
- Training and evaluation write to `outputs/` by default.
- The export script takes its settings from `smoledgevlm/config.py`.
- No change to the model, the training recipe or any reported number. Encoder features, connector outputs and captions
  were checked to be identical to the previous version.

## 2.0.0: retrained on broader data

- R, the merged-path connector and a LoRA adapter were retrained on 127,602 images from COCO, TextCaps and
  VizWiz-Captions, with augmentation: `notebooks/retrain_v2_kaggle.ipynb`.
- A full-encoder check under v2: `notebooks/v2_full_encoder_check_kaggle.ipynb`.
- v2 weights (R and connector) and v2 results: T4 comparison, and a board run on the 100-image comparison set.
- `scripts/score_board.py` scores a board run against the compared models.
- `jetson/live_caption.py` is a live camera test. The board runtime also loads packages written by the v2 notebook.
- The project page is in `docs/` (GitHub Pages).

## 1.0.0: first release (2026-10-14)

- The method: token merging after SigLIP layer 3, restoration module R, size bias, unmerge, pooling connector.
- Training (two stages) and evaluation (CIDEr-D, CLIPScore, paired bootstrap) on fixed COCO 2014 splits.
- The Jetson Nano 2 GB runtime, export script and benchmark.
- Results: T4 with 3 seeds, the Jetson at 5 W and 10 W, inside SmolVLM-256M, and against released compact VLMs.
