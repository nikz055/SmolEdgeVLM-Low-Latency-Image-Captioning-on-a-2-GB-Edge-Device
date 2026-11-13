# Running SmolEdgeVLM on a Jetson Nano 2 GB

The device runs JetPack 4.6.1 (Python 3.6, CUDA 10.2, PyTorch 1.10), which current `transformers` does not support,
so the model runs from a plain-PyTorch re-implementation with exported float16 weights.

| File | Where it runs | What it does |
|---|---|---|
| `export_for_jetson.py` | PC or Kaggle (with `transformers`) | Packs SigLIP, SmolLM2, R and both connectors into `smoledgevlm_jetson_weights.npz` (float16) and `smoledgevlm_jetson_meta.json` (sizes, prompt ids, tokenizer byte table, optional reference captions) |
| `smoledgevlm_jetson.py` | Jetson | The model: encoder (full or merged), connector, SmolLM2 with KV cache, greedy decoding. Needs only PyTorch, NumPy and Pillow |
| `benchmark.py` | Jetson | Timing with a timer between stages, captions for scoring, a sustained run, and sensor logging |

## 1. Export (on a PC)

```bash
python jetson/export_for_jetson.py --weights-dir weights/v1 --out jetson_package --ref-images my_images --ref-count 10
```

## 2. Prepare the board

```bash
sudo systemctl set-default multi-user.target && sudo reboot   # text console: frees about 1 GB of shared memory
sudo nvpmodel -m 0                                             # 10 W (use -m 1 for 5 W)
sudo dmesg -n 1                                                # keep kernel messages off the console
```

Copy `smoledgevlm_jetson.py`, `benchmark.py`, the two files from `jetson_package/` and an `images/` folder into one
folder on the board.

## 3. Run

```bash
python3 smoledgevlm_jetson.py --images images --runs 3     # caption and time each image with both pipelines
python3 benchmark.py --quick                                # 5-minute check
python3 benchmark.py --images images_test500                # full run: about 2 hours at 10 W
python3 benchmark.py --images images_test500 --resume       # continue after an interruption
```

`benchmark.py` writes `per_image.csv`, `stages.csv`, `sustained.csv`, `sensors.csv`, `tegrastats.log` and
`summary.json`, and ends with a result block like the one in `docs/static/jetson_result.jpg`.

### Live camera test

`live_caption.py` captions what a USB or CSI camera sees (or a folder of photos), prints the timing of each caption,
saves every frame, and can ask for a rating after each caption (`c` correct, `p` partly correct, `w` wrong, add `h` if
it names something that is not there).

```bash
python3 live_caption.py --weights smoledgevlm_jetson_weights.npz --meta smoledgevlm_jetson_meta.json --camera usb --rate --out live
python3 live_caption.py --weights smoledgevlm_jetson_weights.npz --meta smoledgevlm_jetson_meta.json --camera usb --every 5 --count 20 --out demo
```

Output: `frames/`, `live_log.csv` and `live_summary.json`.

### Version 2 package

`notebooks/retrain_v2_kaggle.ipynb` writes a board package (`smoledgevlm_v2_lora_weights.npz`, `smoledgevlm_v2_meta.json`)
that `smoledgevlm_jetson.py`, `benchmark.py` and `live_caption.py` load directly with `--weights` and `--meta`. In that
package only the merged path is retrained; its full-encoder path still uses the original connector.

To record a run for a video: `script -t 2>timing.log output.log`, run the benchmark, `exit`, then replay with
`scriptreplay timing.log output.log 20` (20× speed).

## Measured on the device (10 W, 500 COCO test images)

| | Full encoder | SmolEdgeVLM |
|---|---|---|
| Time to first token | 2.52 s | 1.46 s (−42%) |
| Whole caption | 4.51 s | 3.36 s (−26%) |
| Layers 4-12 | 1,325 ms | 251 ms |
| Captions per minute (5 min sustained) | 12.5 | 17.3 |
| Board RAM in use | 1,915 MB | 1,912 MB |
