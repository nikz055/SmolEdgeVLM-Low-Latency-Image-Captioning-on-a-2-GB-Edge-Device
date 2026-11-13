# Installation and usage

## Requirements

| Use | Hardware | Software |
|---|---|---|
| Captioning, evaluation, training | Any PC; a CUDA GPU for training (a 16 GB Tesla T4 was used) | Python ≥ 3.9, PyTorch ≥ 1.12, transformers ≥ 4.40 (checked with 5.14; the notebooks pin 4.57) |
| Board scoring (`scripts/score_board.py`) | Any PC | `pycocoevalcap` and Java (for METEOR and the PTB tokenizer) |
| Retraining v2 (notebooks) | Kaggle, one Tesla T4 | Installed by the notebook |
| Deployment | Jetson Nano 2 GB | JetPack 4.6.1 (Python 3.6, CUDA 10.2), PyTorch 1.10, NumPy, Pillow |

The pretrained SigLIP-B/16@512 and SmolLM2-135M weights (about 1.1 GB) download from Hugging Face the first time they
are used. The trained R and connectors are in `weights/` (7 MB per version).

## Install (PC)

```bash
git clone https://github.com/nikz055/SmolEdgeVLM-Low-Latency-Image-Captioning-on-a-2-GB-Edge-Device.git
cd SmolEdgeVLM-Low-Latency-Image-Captioning-on-a-2-GB-Edge-Device
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                # add ",eval" for scripts/score_board.py
```

`pip install -r requirements.txt` installs the runtime dependencies only. Install PyTorch first from
[pytorch.org](https://pytorch.org) if you need a specific CUDA build.

Check the install:

```bash
pytest                                  # 7 CPU tests, no downloads
python scripts/caption.py docs/static/coco_4754.jpg
```

With `transformers` 5.x, loading SigLIP prints a report that lists the `text_model.*` weights as UNEXPECTED. This is
expected: only the vision tower is used.

## Caption images

```bash
python scripts/caption.py photo.jpg                          # v1: full encoder and SmolEdgeVLM, with timings
python scripts/caption.py photo.jpg --weights weights/v2     # v2 (merged path only; no LoRA)
python scripts/caption.py photo.jpg --merge-map merge.png    # also save a picture of the 256 token groups
```

From Python:

```python
from PIL import Image
from smoledgevlm import SmolEdgeVLM

model = SmolEdgeVLM('weights/v1')                  # device and dtype are optional
result = model.caption(Image.open('photo.jpg'))    # mode='smoledgevlm' (default) or 'full'
print(result['caption'], result['ttft_ms'])
```

`model.encode(pixels, mode)` returns the encoder features for `'full'`, `'smoledgevlm'` or `'merge_no_R'`.

## Train and evaluate (v1 recipe)

Put MS-COCO 2014 in one folder: `train2014/`, `val2014/` and `annotations/captions_{train,val}2014.json`. The splits are
fixed in [`smoledgevlm/data.py`](../smoledgevlm/data.py). Outputs go to `outputs/`, which git ignores, so the
released weights are never overwritten.

```bash
python scripts/train_restoration.py --coco /data/coco2014                                   # stage 1: outputs/restore_R.pt
python scripts/train_connector.py   --coco /data/coco2014 --mode smoledgevlm --weights outputs --out outputs/connector_smoledgevlm.pt
python scripts/train_connector.py   --coco /data/coco2014 --mode full                          --out outputs/connector_full_encoder.pt
python scripts/evaluate.py          --coco /data/coco2014 --weights outputs --modes full,smoledgevlm
```

`evaluate.py` reports CIDEr-D and CLIPScore on the 500 test images, the median TTFT, and paired bootstrap intervals
against the full encoder. To evaluate the released weights, leave out `--weights`.

The v2 recipe (more data, augmentation, LoRA) is in [`notebooks/`](../notebooks/README.md).

## Score a Jetson board run

```bash
python scripts/score_board.py \
    --comparison results/jetson_board_100/captions_all_models.csv \
    --run path/to/per_image.csv --name "my run" --out outputs/scores.json
```

## Deploy on a Jetson Nano

Export on a PC, then copy the board scripts and the package to the board: [jetson/README.md](../jetson/README.md).

## Development

- `smoledgevlm/` is the reference implementation, and all scripts import it. Keep `config.py` as the single place for
  model settings.
- The files in `jetson/` other than `export_for_jetson.py` run on the board, so they must stay compatible with
  Python 3.6 and PyTorch 1.10, with no `transformers` and no f-strings.
- If you change the model maths, change `model.py` and `jetson/smoledgevlm_jetson.py` together. Check that the
  exported package still reproduces the reference captions (`export_for_jetson.py --ref-images`, then the
  `matches ref` column on the board).
- The notebooks are self-contained records of the v2 runs: they write their own copies of the board code, under the
  working name `jetson_foremerge.py`. Do not edit them to change results.
- Run `pytest` before committing. Put generated files in `outputs/`.
