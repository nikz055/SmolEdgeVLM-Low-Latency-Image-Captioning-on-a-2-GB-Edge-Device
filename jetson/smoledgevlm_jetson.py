#!/usr/bin/env python3
"""SmolEdgeVLM runtime for the Jetson Nano 2 GB: full encoder vs SmolEdgeVLM, captioned and timed.

Needs only Python 3.6+, PyTorch 1.10+, NumPy and Pillow (no transformers). Same maths as smoledgevlm/model.py:
frozen SigLIP-B/16@512, merge 1024 -> 256 tokens after layer 3 (fixed lattice, 7 x 7 window), restoration
module R, layers 4-12 with a log(size) attention bias, unmerge, pooling connector, frozen SmolLM2-135M with a
KV cache and greedy decoding.

    python3 smoledgevlm_jetson.py --images images
    python3 smoledgevlm_jetson.py --images images --runs 3 --dtype float16 --out jetson_results

Inputs (same folder by default): smoledgevlm_jetson_weights.npz and smoledgevlm_jetson_meta.json, made on a PC
with export_for_jetson.py. Outputs: <out>.csv (one row per image and pipeline) and <out>.json (summary).
"""
import argparse, glob, json, math, os, resource, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ------------------------------------------------------------------ loading
torch.backends.cudnn.enabled = False                # nothing here needs cuDNN; not loading it saves memory


def load_model(weights, meta_path, device, dtype):
    meta = json.load(open(meta_path))
    W = {}
    z = np.load(weights)
    for k in z.files:                              # one tensor at a time: keeps peak memory low on 2 GB
        W[k] = torch.from_numpy(z[k]).to(device=device, dtype=dtype)
    z.close()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return meta, W


def mem_mb():
    """Peak process memory (MB) and currently available system memory (MB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    avail = float('nan')
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                avail = int(line.split()[1]) / 1024.0
    except IOError:
        pass
    return peak, avail


# ------------------------------------------------------------------ building blocks
def gelu_tanh(x):                                  # SigLIP's gelu_pytorch_tanh (F.gelu has no "approximate" in torch 1.10)
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def rms_norm(x, w, eps):                           # LlamaRMSNorm: normalise in float32, scale in the model dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


class SmolEdgeVLMJetson(object):
    def __init__(self, weights, meta_path, device='cuda', dtype=torch.float16):
        self.device, self.dtype = device, dtype
        self.meta, self.W = load_model(weights, meta_path, device, dtype)
        v, l, fm = self.meta['vision'], self.meta['lm'], self.meta['merge']
        self.v, self.l = v, l
        self.k, self.M, self.radius = fm['k'], fm['M'], fm['radius']
        self.grid = v['image_size'] // v['patch']
        self.N = self.grid * self.grid
        idx = np.arange(self.N)
        rows, cols = idx // self.grid, idx % self.grid
        self.coords = torch.from_numpy(np.stack([rows, cols], 1)).to(device)
        lattice = np.where((rows % 2 == 0) & (cols % 2 == 0), 2.0, np.where((rows % 2 == 1) & (cols % 2 == 1), 1.0, 0.0))
        score = torch.from_numpy(lattice - 1e-6 * idx)
        self.anchors = torch.topk(score, self.M).indices.sort().values.to(device)
        is_anc = torch.zeros(self.N, dtype=torch.bool, device=device)
        is_anc[self.anchors] = True
        self.others = (~is_anc).nonzero().squeeze(1)
        dist = (self.coords[self.others][:, None, :] - self.coords[self.anchors][None, :, :]).abs().max(-1)[0]
        self.outside = dist > self.radius                       # [768, 256]: anchors a patch may not join
        hd = l['hidden'] // l['heads']
        self.hd = hd
        self.inv_freq = 1.0 / (l['rope_theta'] ** (torch.arange(0, hd, 2, device=device).float() / hd))
        self.prompt = self.W['lm.embed'][torch.tensor(self.meta['prompt_ids'], device=device)][None]
        self.eos = self.meta['eos_id']
        self.special = set(self.meta['special_ids'])
        self.id_bytes = [bytes.fromhex(h) for h in self.meta['id_bytes_hex']]
        mean = torch.tensor(v['mean']).view(1, 3, 1, 1)
        std = torch.tensor(v['std']).view(1, 3, 1, 1)
        self.mean, self.std = mean.to(device), std.to(device)

    # ---------------------------------------------------------- image
    def pixels(self, img):
        """SigLIP preprocessing: bicubic resize to 512 x 512, scale to [0, 1], normalise with mean/std 0.5."""
        h, w = self.v['resize']
        x = np.asarray(img.convert('RGB').resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
        t = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)
        return ((t - self.mean) / self.std).to(self.dtype)

    # ---------------------------------------------------------- SigLIP
    def ln(self, x, p, eps):
        return F.layer_norm(x, (x.shape[-1],), self.W[p + '.w'], self.W[p + '.b'], eps)

    def embed(self, px):
        """Patch embedding. A 16 x 16 convolution with stride 16 equals cutting the image into 16 x 16 squares and
        applying one linear layer, which is what we do: same numbers, but no cuDNN (cuDNN alone can take
        several hundred MB on a Jetson Nano)."""
        W, P, g = self.W, self.v['patch'], self.grid
        B, C = px.shape[0], px.shape[1]
        x = px.reshape(B, C, g, P, g, P).permute(0, 2, 4, 1, 3, 5).reshape(B, g * g, C * P * P)   # [1, 1024, 768]
        w = W['v.patch.w'].reshape(W['v.patch.w'].shape[0], -1)                                      # [768, 3*16*16]
        return F.linear(x, w, W['v.patch.b']) + W['v.pos'][None]                                     # [1, 1024, 768]

    def vit_layer(self, i, h, bias=None):
        W, p, eps = self.W, 'v.L%d.' % i, self.v['ln_eps']
        B, N, D = h.shape
        H = self.v['heads']
        hd = D // H
        x = self.ln(h, p + 'ln1', eps)
        q = F.linear(x, W[p + 'q.w'], W[p + 'q.b']).view(B, N, H, hd).transpose(1, 2)
        k = F.linear(x, W[p + 'k.w'], W[p + 'k.b']).view(B, N, H, hd).transpose(1, 2)
        v = F.linear(x, W[p + 'v.w'], W[p + 'v.b']).view(B, N, H, hd).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-1, -2)) * (hd ** -0.5)                    # [1, 12, N, N]
        if bias is not None:
            s = s + bias                                                              # log(size) of each key token
        a = torch.softmax(s.float(), dim=-1).to(q.dtype)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, D)
        h = h + F.linear(o, W[p + 'o.w'], W[p + 'o.b'])
        x = self.ln(h, p + 'ln2', eps)
        return h + F.linear(gelu_tanh(F.linear(x, W[p + 'fc1.w'], W[p + 'fc1.b'])), W[p + 'fc2.w'], W[p + 'fc2.b'])

    def merge(self, h):
        """Every non-anchor patch joins its most similar anchor within the 7 x 7 window; average each group."""
        a, o = self.anchors, self.others
        hn = F.normalize(h[0].float(), dim=-1)
        sim = hn[o] @ hn[a].t()                                                       # [768, 256] cosine
        local = sim.masked_fill(self.outside, float('-inf'))
        assign = torch.empty(self.N, dtype=torch.long, device=self.device)
        assign[a] = torch.arange(self.M, device=self.device)
        assign[o] = local.argmax(1)
        x = h[0].float()
        ones = torch.ones(self.N, device=self.device)
        sizes = torch.zeros(self.M, device=self.device).index_add_(0, assign, ones)
        mean = torch.zeros(self.M, x.shape[-1], device=self.device).index_add_(0, assign, x) / sizes[:, None]
        return mean[None].to(self.dtype), sizes, assign

    def restore(self, m):
        """R: m' = m + W2 GELU(W1 LN(m))."""
        W = self.W
        x = F.layer_norm(m, (m.shape[-1],), W['r.norm.w'], W['r.norm.b'], 1e-5)
        y = F.gelu(F.linear(x, W['r.fc1.w'], W['r.fc1.b']))
        return m + F.linear(y, W['r.fc2.w'], W['r.fc2.b'])

    def encode(self, px, mode):
        """Vision tokens on the 32 x 32 grid, [1, 1024, 768]. mode: 'full' or 'smoledgevlm'."""
        h = self.embed(px)
        L = self.v['layers']
        if mode == 'full':
            for i in range(L):
                h = self.vit_layer(i, h)
            return self.ln(h, 'v.post', self.v['ln_eps'])
        for i in range(self.k):
            h = self.vit_layer(i, h)
        m, sizes, assign = self.merge(h)
        m = self.restore(m)
        bias = torch.log(sizes)[None, None, None, :].to(self.dtype)
        for i in range(self.k, L):
            m = self.vit_layer(i, m, bias)
        return self.ln(m, 'v.post', self.v['ln_eps'])[:, assign, :]

    # ---------------------------------------------------------- connector
    def connector(self, feats, which):
        W, p = self.W, ('cf.' if which == 'full' else 'cm.')
        B, n, d = feats.shape
        s = int(round(n ** 0.5))
        x = F.adaptive_avg_pool2d(feats.transpose(1, 2).reshape(B, d, s, s), (16, 16))
        x = x.flatten(2).transpose(1, 2)                                              # [1, 256, 768]
        return F.linear(F.gelu(F.linear(x, W[p + 'fc1.w'], W[p + 'fc1.b'])), W[p + 'fc2.w'], W[p + 'fc2.b'])

    # ---------------------------------------------------------- SmolLM2 (Llama) with KV cache
    def rope(self, pos0, T):
        pos = torch.arange(pos0, pos0 + T, device=self.device).float()
        f = pos[:, None] * self.inv_freq[None, :]
        e = torch.cat([f, f], -1)
        return e.cos().to(self.dtype)[None, None], e.sin().to(self.dtype)[None, None]

    @staticmethod
    def rot(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], -1)

    def lm_step(self, x, cache, pos0):
        """x: [1, T, 576] input embeddings. Returns logits of the last position and the updated cache."""
        W, l = self.W, self.l
        B, T, D = x.shape
        Hq, Hk, hd = l['heads'], l['kv_heads'], self.hd
        rep = Hq // Hk
        cos, sin = self.rope(pos0, T)
        mask = None
        if T > 1:
            mask = torch.full((T, T), float('-inf'), device=self.device).triu(1).to(self.dtype)
        h = x
        for i in range(l['layers']):
            p = 'lm.L%d.' % i
            a = rms_norm(h, W[p + 'ln_in'], l['rms_eps'])
            q = F.linear(a, W[p + 'q']).view(B, T, Hq, hd).transpose(1, 2)
            k = F.linear(a, W[p + 'k']).view(B, T, Hk, hd).transpose(1, 2)
            v = F.linear(a, W[p + 'v']).view(B, T, Hk, hd).transpose(1, 2)
            q = q * cos + self.rot(q) * sin
            k = k * cos + self.rot(k) * sin
            if cache[i] is not None:
                k = torch.cat([cache[i][0], k], 2)
                v = torch.cat([cache[i][1], v], 2)
            cache[i] = (k, v)
            kk, vv = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
            s = torch.matmul(q, kk.transpose(-1, -2)) * (hd ** -0.5)
            if mask is not None:
                s = s + mask
            att = torch.softmax(s.float(), dim=-1).to(q.dtype)
            o = torch.matmul(att, vv).transpose(1, 2).reshape(B, T, D)
            h = h + F.linear(o, W[p + 'o'])
            a = rms_norm(h, W[p + 'ln_post'], l['rms_eps'])
            h = h + F.linear(F.silu(F.linear(a, W[p + 'gate'])) * F.linear(a, W[p + 'up']), W[p + 'down'])
        last = rms_norm(h[:, -1:], W['lm.norm'], l['rms_eps'])
        head = W['lm.head'] if 'lm.head' in W else W['lm.embed']
        return F.linear(last, head)[:, -1], cache                                   # only the last position is needed

    def decode(self, ids):
        s = b''.join(self.id_bytes[i] for i in ids if i not in self.special).decode('utf-8', errors='replace')
        if self.meta.get('clean_up_spaces'):
            for a, b in ((' .', '.'), (' ?', '?'), (' !', '!'), (' ,', ','), (" ' ", "'"), (" n't", "n't"),
                         (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")):
                s = s.replace(a, b)
        return s.strip()

    def sync(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()

    def caption(self, px, mode, max_new=24):
        """One timed caption. Returns caption, ms to the first token, ms for the whole caption, token count."""
        with torch.no_grad():
            self.sync()
            t0 = time.perf_counter()
            Z = self.connector(self.encode(px, mode), mode)
            seq = torch.cat([Z, self.prompt], 1)
            cache = [None] * self.l['layers']
            logits, cache = self.lm_step(seq, cache, 0)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError('non-finite values (float16 overflow?): rerun with --dtype float32')
            nxt = int(logits.argmax(-1))
            self.sync()
            ttft = (time.perf_counter() - t0) * 1e3
            toks, pos = [], seq.shape[1]
            for _ in range(max_new):
                if nxt == self.eos:
                    break
                toks.append(nxt)
                e = self.W['lm.embed'][nxt][None, None]
                logits, cache = self.lm_step(e, cache, pos)
                pos += 1
                nxt = int(logits.argmax(-1))
            self.sync()
            total = (time.perf_counter() - t0) * 1e3
        return self.decode(toks), ttft, total, len(toks)


# ------------------------------------------------------------------ benchmark
def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(q / 100.0 * len(xs))) - 1)] if xs else float('nan')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=os.path.join(here, 'smoledgevlm_jetson_weights.npz'))
    ap.add_argument('--meta', default=os.path.join(here, 'smoledgevlm_jetson_meta.json'))
    ap.add_argument('--images', default=os.path.join(here, 'images'))
    ap.add_argument('--limit', type=int, default=0, help='use only the first N images (0 = all)')
    ap.add_argument('--runs', type=int, default=3, help='timed runs per image and pipeline (median is kept)')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--max-new', type=int, default=24)
    ap.add_argument('--dtype', default='float16', choices=['float16', 'float32'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--modes', default='full,smoledgevlm')
    ap.add_argument('--out', default=os.path.join(here, 'jetson_results'))
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(',') if m.strip()]
    dtype = torch.float16 if a.dtype == 'float16' else torch.float32

    files = sorted(f for f in glob.glob(os.path.join(a.images, '*')) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit('no .jpg/.png images found in %s' % a.images)
    _, avail0 = mem_mb()
    print('torch %s | device %s | %s | %d images | %d runs each | available memory before loading: %.0f MB'
          % (torch.__version__, a.device, a.dtype, len(files), a.runs, avail0))
    t = time.time()
    fm = SmolEdgeVLMJetson(a.weights, a.meta, a.device, dtype)
    peak, avail = mem_mb()
    print('model loaded in %.1f s | process peak %.0f MB | available now %.0f MB' % (time.time() - t, peak, avail))

    px0 = fm.pixels(Image.open(files[0]))
    for m in modes:
        for _ in range(a.warmup):
            fm.caption(px0, m, a.max_new)
    print('warm-up done')

    rows, ref = [], fm.meta.get('reference', {})
    gpu_peak = dict((m, 0.0) for m in modes)
    for n, f in enumerate(files):
        img = Image.open(f)
        t = time.perf_counter()
        px = fm.pixels(img)
        prep = (time.perf_counter() - t) * 1e3
        res = dict((m, []) for m in modes)
        for _ in range(a.runs):                                   # round-robin: drift hits both pipelines equally
            for m in modes:
                if a.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                res[m].append(fm.caption(px, m, a.max_new))
                if a.device == 'cuda':
                    gpu_peak[m] = max(gpu_peak[m], torch.cuda.max_memory_allocated() / 2 ** 20)
        name = os.path.basename(f)
        for m in modes:
            cap = res[m][0][0]
            row = {'image': name, 'mode': m, 'caption': cap,
                   'ttft_ms': float(np.median([r[1] for r in res[m]])),
                   'total_ms': float(np.median([r[2] for r in res[m]])),
                   'tokens': res[m][0][3], 'prep_ms': prep,
                   'matches_reference': (cap == ref[name][m]) if name in ref and m in ref[name] else ''}
            rows.append(row)
        print('[%d/%d] %s | ' % (n + 1, len(files), name) +
              ' | '.join('%s %.0f ms: %s' % (m, rows[-len(modes) + j]['ttft_ms'], rows[-len(modes) + j]['caption'])
                         for j, m in enumerate(modes)))

    summary = {'device': a.device, 'dtype': a.dtype, 'torch': torch.__version__, 'images': len(files), 'runs': a.runs,
               'max_new': a.max_new, 'available_mb_before_load': avail0, 'process_peak_mb': mem_mb()[0], 'modes': {}}
    for m in modes:
        tt = [r['ttft_ms'] for r in rows if r['mode'] == m]
        to = [r['total_ms'] for r in rows if r['mode'] == m]
        mt = [r['matches_reference'] for r in rows if r['mode'] == m and r['matches_reference'] != '']
        summary['modes'][m] = {'ttft_median_ms': float(np.median(tt)), 'ttft_p95_ms': pct(tt, 95),
                               'total_median_ms': float(np.median(to)), 'total_p95_ms': pct(to, 95),
                               'gpu_peak_mb': gpu_peak[m] if a.device == 'cuda' else float('nan'),
                               'reference_match': ('%d/%d' % (sum(mt), len(mt))) if mt else 'no reference'}
    if 'full' in summary['modes'] and 'smoledgevlm' in summary['modes']:
        f_, g_ = summary['modes']['full'], summary['modes']['smoledgevlm']
        summary['ttft_change_pct'] = 100.0 * (g_['ttft_median_ms'] / f_['ttft_median_ms'] - 1)
        summary['total_change_pct'] = 100.0 * (g_['total_median_ms'] / f_['total_median_ms'] - 1)

    import csv
    with open(a.out + '.csv', 'w') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump(summary, open(a.out + '.json', 'w'), indent=1)

    print('\n' + '=' * 92)
    print('JETSON RESULT (%s, %s, batch 1, %d images x %d runs, median)' % (a.device, a.dtype, len(files), a.runs))
    print('=' * 92)
    print('%-11s %12s %10s %14s %10s %12s %16s' % ('pipeline', 'TTFT ms', 'p95', 'caption ms', 'p95', 'GPU peak MB', 'matches ref'))
    for m in modes:
        s = summary['modes'][m]
        print('%-11s %12.1f %10.1f %14.1f %10.1f %12.0f %16s' % (m, s['ttft_median_ms'], s['ttft_p95_ms'], s['total_median_ms'],
                                                                s['total_p95_ms'], s['gpu_peak_mb'], s['reference_match']))
    if 'ttft_change_pct' in summary:
        print('SmolEdgeVLM vs full: time to first token %+.1f%%, whole caption %+.1f%%'
              % (summary['ttft_change_pct'], summary['total_change_pct']))
    print('process peak memory %.0f MB | written %s.csv and %s.json' % (summary['process_peak_mb'], a.out, a.out))


if __name__ == '__main__':
    main()
