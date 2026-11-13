# Version 2 weights

`restore_R.pt` and `connector_smoledgevlm.pt` trained on 127,602 images (COCO train2014, TextCaps, VizWiz-Captions),
in the same format as the files in `weights/v1/`, so the merged path loads with:

```python
from smoledgevlm.model import SmolEdgeVLM
model = SmolEdgeVLM(weights_dir='weights/v2')
```

The LoRA adapter used for the board results is merged into the language model inside the board package written by
`notebooks/retrain_v2_kaggle.ipynb`; it is not included here. There is no version 2 full-encoder connector yet.
