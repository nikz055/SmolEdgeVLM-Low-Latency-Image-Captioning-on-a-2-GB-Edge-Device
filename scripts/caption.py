"""Caption one image with the full encoder and with SmolEdgeVLM, and print both with their timings.
(With weights that have no full-encoder connector, such as weights/v2, only SmolEdgeVLM runs.)

    python scripts/caption.py photo.jpg
    python scripts/caption.py photo.jpg --merge-map merge.png      # also save a picture of the token groups
    python scripts/caption.py photo.jpg --weights weights/v2       # version 2 (merged path only)
"""
import argparse

from PIL import Image

import common  # noqa: F401  (adds the repository root to the import path)
from smoledgevlm.model import SmolEdgeVLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--weights', default='weights/v1')
    ap.add_argument('--merge-map', default='')
    a = ap.parse_args()
    model = SmolEdgeVLM(a.weights)
    img = Image.open(a.image).convert('RGB')
    for mode in model.available_modes:
        model.caption(img, mode, max_new=2)                           # warm-up
        r = model.caption(img, mode)
        print('%-12s TTFT %7.1f ms | total %7.1f ms | %s' % (mode, r['ttft_ms'], r['total_ms'], r['caption']))
    if a.merge_map:
        model.merge_map(img).save(a.merge_map)
        print('merge map saved to', a.merge_map)


if __name__ == '__main__':
    main()
