"""Evaluation: CIDEr-D, CLIPScore and paired bootstrap confidence intervals."""
import math
import re
from collections import defaultdict

import numpy as np


def _norm(s):
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', s.lower()).split())


def _ngrams(s, n_max=4):
    w, c = s.split(), defaultdict(float)
    for n in range(1, n_max + 1):
        for i in range(len(w) - n + 1):
            c[(n, ' '.join(w[i:i + n]))] += 1.0
    return c, len(w)


def cider_d(refs, preds, n_max=4, sigma=6.0):
    """CIDEr-D (Vedantam et al., 2015). refs: {id: [captions]}, preds: {id: caption}.
    Returns the corpus score and the per-image scores in the order of preds."""
    ids = list(preds)
    df = defaultdict(float)
    for i in ids:
        seen = set()
        for r in refs[i]:
            seen |= set(_ngrams(_norm(r))[0])
        for g in seen:
            df[g] += 1.0
    log_n = math.log(max(len(ids), 1))

    def vec(counts, length):
        v, nrm = [defaultdict(float) for _ in range(n_max)], [0.0] * n_max
        for (n, g), c in counts.items():
            val = c * (log_n - math.log(max(df[(n, g)], 1.0)))
            v[n - 1][(n, g)] = val
            nrm[n - 1] += val ** 2
        return v, [math.sqrt(x) for x in nrm], length

    scores = []
    for i in ids:
        vc, nc, lc = vec(*_ngrams(_norm(preds[i])))
        acc = np.zeros(n_max)
        for r in refs[i]:
            vr, nr, lr = vec(*_ngrams(_norm(r)))
            for n in range(n_max):
                s = sum(min(vc[n][g], vr[n][g]) * vr[n][g] for g in vc[n])
                if nc[n] > 0 and nr[n] > 0:
                    s /= nc[n] * nr[n]
                acc[n] += s * math.exp(-((lc - lr) ** 2) / (2 * sigma ** 2))
        scores.append(float(np.mean(acc / len(refs[i])) * 10.0))
    return float(np.mean(scores)), scores


class CLIPScorer:
    """CLIPScore (Hessel et al., 2021) = 2.5 * max(cos(image, caption), 0) * 100, CLIP ViT-B/32, no text prefix."""

    def __init__(self, device='cpu', model_id='openai/clip-vit-base-patch32'):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).eval().to(device)
        self.proc = CLIPProcessor.from_pretrained(model_id)

    def __call__(self, images, captions):
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            t = self.proc(text=captions, images=images, return_tensors='pt', padding=True, truncation=True)
            out = self.model(**{k: v.to(self.device) for k, v in t.items()})
            ie, te = F.normalize(out.image_embeds, dim=-1), F.normalize(out.text_embeds, dim=-1)
            return (2.5 * (ie * te).sum(-1).clamp_min(0) * 100).tolist()


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """Mean of a - b over the same images, with a 95% bootstrap confidence interval."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
