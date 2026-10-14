"""Stage 2: train the pooling connector with caption loss through the frozen SmolLM2.

The same recipe is used for every pipeline: 5,000 COCO train images (one caption each), 1,500 steps,
AdamW (lr 1e-3, 40 warm-up steps, cosine decay), batch 16. Encoder features are computed once and cached.

    python scripts/train_connector.py --coco /data/coco2014 --mode smoledgevlm --out weights/connector_smoledgevlm.pt
    python scripts/train_connector.py --coco /data/coco2014 --mode full        --out weights/connector_full_encoder.pt
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from common import lr_cosine, pack_captions
from smoledgevlm import config as C
from smoledgevlm.data import coco_splits
from smoledgevlm.losses import caption_loss
from smoledgevlm.model import SmolEdgeVLM
from smoledgevlm.modules import PoolConnector, pool_grid


def cache_features(model, items, mode, path):
    """Encoder output pooled to 16 x 16, stored once as float16: [len(items), 256, 768]."""
    if os.path.exists(path):
        return np.load(path, mmap_mode='r')
    mm = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(len(items), 256, model.d_v))
    for i, r in enumerate(items):
        with torch.no_grad():
            f = model.encode(model.pixels(Image.open(r['file'])), mode)
            mm[i] = pool_grid(f)[0].half().cpu().numpy()
        if (i + 1) % 500 == 0:
            print('  cached %d/%d' % (i + 1, len(items)))
    mm.flush()
    return np.load(path, mmap_mode='r')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco', required=True)
    ap.add_argument('--mode', choices=['smoledgevlm', 'merge_no_R', 'full'], default='smoledgevlm')
    ap.add_argument('--weights', default='weights', help='folder with restore_R.pt (for the merged modes)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cache', default='features_%s.npy')
    a = ap.parse_args()

    model = SmolEdgeVLM(weights_dir=a.weights if a.mode == 'smoledgevlm' else None)
    items = coco_splits(a.coco)['connector']
    feats = cache_features(model, items, a.mode, a.cache % a.mode)
    caps = [r['caption'] for r in items]

    torch.manual_seed(a.seed)
    conn = PoolConnector(model.d_v, model.d_t, C.CONNECTOR_GRID).to(model.device)
    opt = torch.optim.AdamW(conn.parameters(), lr=a.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(a.seed)
    order, pos, micro, accum = torch.randperm(len(caps), generator=gen).tolist(), 0, 8, 2
    hist, t0 = [], time.time()
    conn.train()
    for step in range(a.steps):
        for g in opt.param_groups:
            g['lr'] = lr_cosine(step, a.steps, a.lr, 40)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            if pos + micro > len(order):
                order, pos = torch.randperm(len(caps), generator=gen).tolist(), 0
            idx = order[pos:pos + micro]
            pos += micro
            f = torch.stack([torch.from_numpy(np.array(feats[j])) for j in idx]).to(model.device, torch.float32)
            ids, mask = pack_captions(model.tok, [caps[j] for j in idx], model.device, C.MAX_CAPTION_TOKENS)
            loss = caption_loss(model.lm, conn(f), model.prompt, ids, mask) / accum
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(conn.parameters(), 1.0)
        opt.step()
        hist.append(total)
        if step == 0 or (step + 1) % 100 == 0:
            print('step %4d/%d  loss %.4f  (%.0f s)' % (step + 1, a.steps, np.mean(hist[-20:]), time.time() - t0))
    torch.save(conn.state_dict(), a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
