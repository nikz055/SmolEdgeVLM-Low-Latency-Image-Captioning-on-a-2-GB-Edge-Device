"""Stage 1: train the restoration module R without captions.

The same image goes through the full frozen encoder (target) and through the merged path
(layers 1-3 -> merge -> R -> layers 4-12 -> unmerge). Only R is updated, to make the two outputs match.
3,000 COCO train images, one pass, batch 8: about 6 minutes on a Tesla T4.

    python scripts/train_restoration.py --coco /data/coco2014 --out weights/restore_R.pt
"""
import argparse
import random
import time

import numpy as np
import torch
from PIL import Image

from common import lr_cosine
from smoledgevlm import config as C
from smoledgevlm.data import coco_splits
from smoledgevlm.losses import restoration_loss
from smoledgevlm.model import SmolEdgeVLM
from smoledgevlm.modules import Restore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco', required=True)
    ap.add_argument('--out', default='weights/restore_R.pt')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    model = SmolEdgeVLM(weights_dir=None)
    imgs = coco_splits(a.coco)['restore']
    R = Restore(model.d_v, C.RESTORE_RANK).to(model.device)
    opt = torch.optim.AdamW(R.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(imgs) // a.batch
    order = random.Random(a.seed).sample(range(len(imgs)), len(imgs))
    hist, t0 = [], time.time()
    R.train()
    for s in range(steps):
        for g in opt.param_groups:
            g['lr'] = lr_cosine(s, steps, a.lr, max(1, steps // 20))
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for j in order[s * a.batch:(s + 1) * a.batch]:
            px = model.pixels(Image.open(imgs[j]['file']))
            with torch.no_grad():
                h = model.layers_to_cut(px)
                target = model.full_from_cut(h)
            loss = restoration_loss(model.merged_from_cut(h, restore=R), target) / a.batch
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0)
        opt.step()
        hist.append(total)
        if s == 0 or (s + 1) % 25 == 0 or s == steps - 1:
            print('step %4d/%d  loss %.4f  (%.0f s)' % (s + 1, steps, np.mean(hist[-20:]), time.time() - t0))
    torch.save(R.state_dict(), a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
