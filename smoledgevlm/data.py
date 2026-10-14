"""MS-COCO 2014 splits used in every experiment.

  connector training   train2014, first 5,000 images (one caption each, lowest annotation id)
  R training           train2014, next 3,000 images (no captions used)
  development          val2014, first 300 images with 5 captions (design decisions only)
  test                 val2014, next 500 images (all reported numbers; never used for a decision)
"""
import json
import os
from collections import defaultdict


def load_captions(annotation_file):
    raw = json.load(open(annotation_file))
    refs = defaultdict(list)
    for a in raw['annotations']:
        refs[a['image_id']].append((a['id'], a['caption'].strip()))
    return refs, {im['id']: im['file_name'] for im in raw['images']}


def coco_splits(coco_root, n_connector=5000, n_restore=3000, n_dev=300, n_test=500):
    """coco_root must contain train2014/, val2014/ and annotations/captions_{train,val}2014.json."""
    trefs, tfiles = load_captions(os.path.join(coco_root, 'annotations', 'captions_train2014.json'))
    train = []
    for i in sorted(trefs):
        f = os.path.join(coco_root, 'train2014', tfiles[i])
        if os.path.exists(f):
            train.append({'image_id': i, 'file': f, 'caption': sorted(trefs[i])[0][1]})
        if len(train) >= n_connector + n_restore:
            break
    vrefs, vfiles = load_captions(os.path.join(coco_root, 'annotations', 'captions_val2014.json'))
    val = []
    for i in sorted(vrefs):
        f = os.path.join(coco_root, 'val2014', vfiles[i])
        caps = [c for _, c in sorted(vrefs[i])]
        if os.path.exists(f) and len(caps) >= 5:
            val.append({'image_id': i, 'file': f, 'captions': caps[:5]})
        if len(val) >= n_dev + n_test:
            break
    splits = {'connector': train[:n_connector], 'restore': train[n_connector:n_connector + n_restore],
              'dev': val[:n_dev], 'test': val[n_dev:n_dev + n_test]}
    assert not ({r['image_id'] for r in splits['dev']} & {r['image_id'] for r in splits['test']})
    return splits
