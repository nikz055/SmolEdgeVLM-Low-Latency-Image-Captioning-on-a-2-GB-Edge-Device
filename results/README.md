# Results files

Every number in [docs/results.md](../docs/results.md) comes from one of these files.

| Folder / file | What it holds |
|---|---|
| `v1/t4_controlled_3_seeds.json` | Tesla T4, 500 COCO test images, 3 seeds: full encoder, merge without R, SmolEdgeVLM, with paired bootstrap intervals |
| `v1/jetson_nano_500_images_10w.json` | Jetson Nano 2 GB at 10 W: 500 images, stage times, sustained run, on-device CIDEr-D, RAM |
| `v1/jetson_nano_20_images_5w.json` | Jetson Nano 2 GB at 5 W, 20 images × 3 runs |
| `v1/smolvlm_256m_cross_model.json` | The merge step and R inside the released SmolVLM-256M (T4) |
| `v1/compact_vlm_comparison_t4.json` | SmolEdgeVLM against released compact VLMs (T4, 500 images) |
| `v1/sample_captions.json` | The captions shown in `docs/static/sample_captions.png` |
| `v2/` | Version 2: T4 comparison and the board run (see [v2/README.md](v2/README.md)) |
| `jetson_board_100/captions_all_models.csv` | Captions of every compared model on the 100 board images, with the five COCO references |
| `jetson_board_100/all_models.csv` | Board timing, memory and scores for every compared model (the source of the board table) |
| `jetson_board_100/bootstrap_vs_ours.csv` | SmolEdgeVLM v1 minus each model: CIDEr, METEOR and ROUGE-L with 95% intervals |
