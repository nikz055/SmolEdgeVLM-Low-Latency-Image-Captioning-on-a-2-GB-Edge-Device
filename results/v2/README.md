# Version 2 results

| File | What it holds |
|---|---|
| `t4_comparison.json` | Output of `notebooks/retrain_v2_kaggle.ipynb`: training time of each part, caption lengths, fair-rule CIDEr and METEOR for the original and version 2 models (with and without LoRA) on 500 images each of COCO held-out, nocaps validation and TextCaps validation, paired bootstrap intervals, and every caption |
| `jetson_board_100/summary.json` | Board run of version 2 + LoRA on the 100 comparison images: medians, stage times, peak memory |
| `jetson_board_100/per_image.csv` | Every caption and its timing from that run |
| `jetson_board_100/stages.csv` | Stage-by-stage timing per image |
| `jetson_board_100/scores.json` | BLEU-4, METEOR, ROUGE-L and CIDEr for the run and every compared model, written by `scripts/score_board.py` |

The board run used `jetson/benchmark.py` at 10 W with `--limit 100 --timing-images 100 --runs 3 --max-new 40`. Only the
merged path (`smoledgevlm`) is version 2; the full-encoder path in that run still uses the original connector, so its
captions do not describe version 2. Peak memory comes from the benchmark script, which reads about 120 MB lower than the
script used for the other models' peak memory.

The compared models' captions are in [../jetson_board_100/captions_all_models.csv](../jetson_board_100/captions_all_models.csv).
