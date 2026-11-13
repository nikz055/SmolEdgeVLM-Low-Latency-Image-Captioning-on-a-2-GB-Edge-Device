# Results

Every number below comes from a file in [`results/`](../results); each table names its source. Unless stated
otherwise, quality is measured on the 500 held-out MS-COCO val2014 test images, which were never used for a design
decision. TTFT is the time to first token.

- [v1: controlled comparison on a Tesla T4](#v1-tesla-t4-controlled-comparison)
- [v1: Jetson Nano 2 GB](#v1-jetson-nano-2-gb)
- [v1: inside SmolVLM-256M](#v1-inside-the-released-smolvlm-256m)
- [v1: against released compact VLMs on a T4](#v1-against-released-compact-vlms-tesla-t4)
- [v2: retrained on broader data](#v2-retrained-on-broader-data)
- [v2: Jetson board comparison](#v2-jetson-board-comparison-100-coco-images)

## v1: Tesla T4, controlled comparison

500 test images, 3 seeds (R and connector retrained per seed), float32, batch 1, greedy, at most 24 new tokens.
Source: [`results/v1/t4_controlled_3_seeds.json`](../results/v1/t4_controlled_3_seeds.json).

| Pipeline | TTFT | CIDEr-D | CLIPScore |
|---|---|---|---|
| Full encoder | 113.2 ms | 0.791 ± 0.065 | 69.0 ± 1.6 |
| Merge without R | 71.3 ms | 0.614 ± 0.014 | 64.9 ± 0.5 |
| **SmolEdgeVLM** (merge + R) | **71.7 ms** | **0.786 ± 0.028** | **69.3 ± 0.5** |

| Paired difference (95% bootstrap CI) | CIDEr-D | CLIPScore | Verdict |
|---|---|---|---|
| SmolEdgeVLM − full encoder | −0.005 [−0.032, +0.024] | +0.35 [−0.04, +0.74] | no significant difference |
| SmolEdgeVLM − merge without R | +0.171 [+0.139, +0.204] | +4.45 [+3.90, +5.07] | R recovers the quality lost by merging |

## v1: Jetson Nano 2 GB

JetPack 4.6.1, PyTorch 1.10, float16, batch 1, board runtime from [`jetson/`](../jetson).
Sources: [`results/v1/jetson_nano_500_images_10w.json`](../results/v1/jetson_nano_500_images_10w.json),
[`results/v1/jetson_nano_20_images_5w.json`](../results/v1/jetson_nano_20_images_5w.json).

| Setting | Full encoder | SmolEdgeVLM | TTFT change | Whole caption change |
|---|---|---|---|---|
| 5 W, 20 images × 3 runs | 3.80 s | 2.21 s | −41.8% | −22.1% |
| **10 W, 500 test images** | **2.52 s** | **1.46 s** | **−42.2%** | **−25.6%** (4.51 → 3.36 s) |

Time per stage, at 10 W (median over 100 images, one synchronised pass per stage):

| Stage | Full encoder | SmolEdgeVLM |
|---|---|---|
| Patch embedding + layers 1–3 | 449 ms | 449 ms |
| Merge + R | | 7.8 + 2.1 ms |
| **Layers 4–12** | **1,325 ms** | **251 ms** (5.3× faster) |
| Post-norm + connector | 8.7 ms | 7.5 ms |
| Language-model prefill (263 tokens) | 744 ms | 744 ms |
| Each further token | 177 ms | 177 ms |

| On the device (10 W, 500 images) | Full encoder | SmolEdgeVLM |
|---|---|---|
| CIDEr-D on the board (T4, same seed) | 0.848 (0.854) | 0.789 (0.792) |
| Captions identical to the T4 run | 466 / 500 | 436 / 500 |
| Captions per minute, 5-minute sustained run | 12.5 | 17.3 |
| Board RAM in use (sustained run, median) | 1,915 MB | 1,912 MB |

A recording of the 500-image run is [`docs/static/jetson_nano_500_image_run.mp4`](static/jetson_nano_500_image_run.mp4).
It is real terminal output replayed at 40× and shows −42.0%. The log uses the method's working name, "foremerge".

## v1: inside the released SmolVLM-256M

Only the step between vision layers 3 and 4 is replaced. R is retrained on SmolVLM's own encoder with 1,000 images and
no captions. Tesla T4, one seed; timing on 50 images, scores on 500.
Source: [`results/v1/smolvlm_256m_cross_model.json`](../results/v1/smolvlm_256m_cross_model.json).

| Mode | TTFT original | TTFT with the merge step | Change | CLIPScore | CIDEr-D |
|---|---|---|---|---|---|
| 13 tiles (default) | 1,301 ms | 579 ms | −55% | 68.8 → 67.3 | 0.188 → 0.256 |
| Single image | 130 ms | 72 ms | −45% | 67.4 → 64.5 | 0.304 → 0.268 |

The latency gain transfers. The quality protection does not fully transfer: the CLIPScore drop is small but
significant, so R needs tuning per model.

## v1: against released compact VLMs (Tesla T4)

Same 500 images and prompt, seed 0, float32, at most 40 new tokens.
Source: [`results/v1/compact_vlm_comparison_t4.json`](../results/v1/compact_vlm_comparison_t4.json).

| Model | Parameters | TTFT | CIDEr-D | CLIPScore |
|---|---|---|---|---|
| **SmolEdgeVLM** | 229 M | **64.7 ms** | 0.792 | 69.3 |
| Full encoder (same model, no merge) | 229 M | 97.6 ms | 0.854 | 70.2 |
| LFM2.5-VL-450M | 450 M | 112.7 ms | 0.723 | 81.6 |
| FastVLM-0.5B | 0.5 B | 308.6 ms | 0.052 | 78.2 |
| Qwen3-VL-2B | 2 B | 392.8 ms | 0.133 | 84.8 |
| SmolVLM-256M | 256 M | 1,013.9 ms | 0.188 | 68.8 |

This is a latency comparison. CIDEr-D favours short COCO-style captions like ours, while the larger models score higher
on CLIPScore.

## v2: retrained on broader data

R, the merged-path connector and a LoRA adapter on SmolLM2 were retrained on 127,602 images: COCO train2014 (82,783),
TextCaps (21,953) and VizWiz-Captions (22,866), with augmented copies in 30% of the steps. The encoder, the language
model and the merge are unchanged. This is one training run (seed 0), from
[`notebooks/retrain_v2_kaggle.ipynb`](../notebooks/retrain_v2_kaggle.ipynb).

Tesla T4, board code in float16, 500 unseen images per set. CIDEr uses the fair rule: first sentence only, with stock
openings such as "In this image we can see" removed.
Source: [`results/v2/t4_comparison.json`](../results/v2/t4_comparison.json).

| Test set | v1 | v2 | v1 + LoRA | v2 + LoRA | v2 + LoRA − v1 + LoRA (95% CI) |
|---|---|---|---|---|---|
| COCO held-out | 0.788 | 0.975 | 0.935 | **1.082** | +0.15 [+0.10, +0.20] |
| nocaps validation | 0.418 | 0.615 | 0.496 | **0.688** | +0.19 [+0.15, +0.23] |
| TextCaps validation | 0.161 | 0.351 | 0.223 | **0.402** | +0.18 [+0.15, +0.21] |

COCO train is now part of the training data, so nocaps is the fairer test of quality.

## v2: Jetson board comparison (100 COCO images)

All models captioned the same 100 COCO val2014 images on the Jetson Nano 2 GB at 10 W (greedy, 40-token cap, medians
over the images). Scores use the fair rule against the five COCO references.
Sources: [`results/jetson_board_100/all_models.csv`](../results/jetson_board_100/all_models.csv) (every compared
model), [`results/v2/jetson_board_100/`](../results/v2/jetson_board_100) (the v2 + LoRA run, scored with
[`scripts/score_board.py`](../scripts/score_board.py)).

| Model | Weights | TTFT | Whole caption | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|---|
| **SmolEdgeVLM v2 + LoRA** | 16-bit | 1.50 s | 3.62 s | 0.326 | 0.263 | 0.566 | 1.183 |
| SmolEdgeVLM v1 + LoRA (COCO only) | 16-bit | 1.46 s | 3.50 s | 0.277 | 0.251 | 0.535 | 1.011 |
| SmolEdgeVLM v1 | 16-bit | 1.46 s | 3.32 s | 0.211 | 0.222 | 0.490 | 0.818 |
| Same model, full encoder | 16-bit | 2.53 s | 4.43 s | 0.222 | 0.234 | 0.503 | 0.858 |
| LFM2-VL-450M | 8-bit | 3.35 s | 12.20 s | 0.296 | 0.295 | 0.563 | 1.158 |
| SmolVLM-500M | 8-bit | 3.28 s | 15.59 s | 0.234 | 0.249 | 0.500 | 0.896 |
| GIT-base-COCO (COCO captioning specialist) | 16-bit | 0.44 s | 0.68 s | 0.457 | 0.322 | 0.629 | 1.590 |

CIDEr difference, v2 + LoRA minus each model (95% paired bootstrap CI, 10,000 resamples), from
[`results/v2/jetson_board_100/scores.json`](../results/v2/jetson_board_100/scores.json):

| Compared with | CIDEr difference | Reading |
|---|---|---|
| LFM2-VL-450M | +0.03 [−0.10, +0.15] | tie, at 2.2× lower TTFT and 3.4× lower caption time |
| SmolEdgeVLM v1 + LoRA | +0.17 [+0.06, +0.30] | v2 better |
| GIT-base-COCO | −0.41 [−0.56, −0.26] | GIT better |

GIT-base-COCO is a single-task model fine-tuned for COCO captions. On this COCO test it beats every general VLM here,
SmolEdgeVLM included, on both speed and quality. SmolEdgeVLM's advantage is over general-purpose compact VLMs of
similar size. The full table, with SmolVLM2, the 8-bit variants and the merge step applied to SmolVLM, is in
`all_models.csv`.

Notes:
- The v2 + LoRA run is a separate board session from the other rows. It uses the same images, settings and scorer.
- In that run only the merged path is v2. Its full-encoder path still used the v1 connector, so it is not reported.
- `bootstrap_vs_ours.csv` gives the same intervals for v1 (SmolEdgeVLM v1 minus each model).
