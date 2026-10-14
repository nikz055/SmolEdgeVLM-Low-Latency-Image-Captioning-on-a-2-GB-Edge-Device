"""Evaluate one or more pipelines on the 500 held-out COCO test images.

Reports CIDEr-D and CLIPScore on all 500 images and the time to first token (TTFT) as the median over the first
--timed images (warm-up first, GPU-synchronised, pipelines alternated). Paired bootstrap CIs compare each mode
with the full encoder.

    python scripts/evaluate.py --coco /data/coco2014 --modes full,merge_no_R,smoledgevlm --out results/eval.json
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image

import common  # noqa: F401  (adds the repository root to the import path)
from smoledgevlm.data import coco_splits
from smoledgevlm.metrics import CLIPScorer, cider_d, paired_bootstrap
from smoledgevlm.model import SmolEdgeVLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco', required=True)
    ap.add_argument('--weights', default='weights')
    ap.add_argument('--modes', default='full,smoledgevlm')
    ap.add_argument('--timed', type=int, default=30)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--out', default='results/eval.json')
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(',')]

    model = SmolEdgeVLM(a.weights)
    if 'merge_no_R' in modes and 'merge_no_R' not in model.connector:
        print('note: merge_no_R uses the SmolEdgeVLM connector unless you pass one trained on merge-without-R features')
    test = coco_splits(a.coco)['test']
    refs = {r['image_id']: r['captions'] for r in test}
    imgs = {r['image_id']: Image.open(r['file']).convert('RGB') for r in test}

    for m in modes:                                              # warm-up
        model.caption(imgs[test[0]['image_id']], m, max_new=2)
    ttft = {m: [] for m in modes}
    for r in test[:a.timed]:
        runs = {m: [] for m in modes}
        for _ in range(a.runs):
            for m in modes:                                      # alternate so drift affects every pipeline equally
                runs[m].append(model.caption(imgs[r['image_id']], m)['ttft_ms'])
        for m in modes:
            ttft[m].append(float(np.median(runs[m])))

    clip = CLIPScorer(model.device)
    res, per = {}, {}
    for m in modes:
        preds = {iid: model.caption(img, m)['caption'] for iid, img in imgs.items()}
        cid, cid_list = cider_d(refs, preds)
        ids = list(preds)
        cs = []
        for s in range(0, len(ids), 64):
            cs += clip([imgs[i] for i in ids[s:s + 64]], [preds[i] for i in ids[s:s + 64]])
        per[m] = {'cider': cid_list, 'clip': cs}
        res[m] = {'cider_d': cid, 'clipscore': float(np.mean(cs)), 'ttft_median_ms': float(np.median(ttft[m])),
                  'samples': [preds[i] for i in ids[:5]]}
        print('%-12s CIDEr-D %.3f  CLIPScore %.2f  TTFT %.1f ms' % (m, cid, res[m]['clipscore'], res[m]['ttft_median_ms']))
    if 'full' in modes:
        for m in modes:
            if m != 'full':
                res[m]['vs_full'] = {k: paired_bootstrap(per[m][k], per['full'][k]) for k in ('cider', 'clip')}
                print('%-12s vs full: CIDEr-D %+.4f [%+.4f, %+.4f], CLIPScore %+.2f [%+.2f, %+.2f]'
                      % ((m,) + tuple(res[m]['vs_full']['cider']) + tuple(res[m]['vs_full']['clip'])))
    json.dump(res, open(a.out, 'w'), indent=2)
    print('written', a.out)


if __name__ == '__main__':
    main()
