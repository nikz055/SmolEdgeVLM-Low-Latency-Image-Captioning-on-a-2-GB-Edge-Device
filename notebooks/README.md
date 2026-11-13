# Version 2 notebooks (Kaggle, one Tesla T4)

Both notebooks are self-contained: they install what they need, write their own copy of the board code, and stop
cleanly before Kaggle's time limit. If a session ends early, add the notebook's previous output as an input and run all
again; finished stages are loaded and the unfinished one continues from its last checkpoint.

## retrain_v2_kaggle.ipynb

Retrains R, the connector and a LoRA adapter on 127,602 images (COCO train2014, TextCaps train, VizWiz-Captions train),
with darkened, blurred, noisy and compressed copies in 30% of the steps. The encoder, the language model and the merge
stay frozen.

- **Inputs:** the COCO 2014 dataset (`jeffaudi/coco-2014-dataset-for-yolov3`) and a dataset with the original seed 0
  weights. TextCaps and nocaps come from Hugging Face, VizWiz from vizwiz.cs.colorado.edu (Internet on).
- **Time:** about 1 hour of data preparation, then R (5 min), connector (75 min) and LoRA (104 min).
- **Output:** `retrain_check.json` (scores on COCO held-out, nocaps and TextCaps with paired bootstrap intervals, all
  captions), the board package `smoledgevlm_v2_jetson.zip`, and the checkpoints in `ckpt/`.

Results: [../results/v2/t4_comparison.json](../results/v2/t4_comparison.json).

## v2_full_encoder_check_kaggle.ipynb

Measures the merge cost under version 2: it trains one connector on the full encoder with the same images, captions,
image changes, order, seed and starting weights as the version 2 connector, then compares four paths (version 2,
version 2 full encoder, original, original full encoder) on the same 1,500 images. No LoRA.

- **Inputs:** the same two datasets, plus a dataset with `r_v2.pt`, `conn_final.pt` and `train_ids.json` from the
  retraining output.
- **Time:** about 4 hours.
- **Output:** `fullconn_check.json`.
