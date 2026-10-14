#!/usr/bin/env python3
"""SmolEdgeVLM on the Jetson: stage timing, captions for quality scoring, sensors and a sustained run.

Uses smoledgevlm_jetson.py (same folder) for all model maths; nothing in the model is changed here.

Phases (one command runs all of them):
  1. Timing   first --timing-images images: clean TTFT and whole-caption time (--runs per pipeline, alternating),
              then one extra pass with a timer between every stage (encoder parts, connector, prefill, each token).
  2. Captions remaining images: one run per pipeline, to collect captions for CIDEr-D / CLIPScore scoring.
  3. Idle     30 s with nothing running (baseline temperature and power).
  4. Sustained --sustain-min minutes of continuous captioning per pipeline (SmolEdgeVLM, then full).
Sensors (temperatures, CPU/GPU clocks, RAM, power rails when the board has them) are logged every second.

    python3 benchmark.py --quick                   # 5-minute check that everything works
    python3 benchmark.py                           # the full run (about 2 hours with 500 images at 10 W)
    python3 benchmark.py --resume                  # continue after an interruption

Outputs in --out (default results): per_image.csv, stages.csv, sustained.csv, sensors.csv, tegrastats.log,
summary.json. Python 3.6 / PyTorch 1.10 compatible.
"""
import argparse, csv, glob, json, os, subprocess, sys, threading, time
from collections import OrderedDict
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoledgevlm_jetson as jf


# ------------------------------------------------------------------ sensors
def _read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except (IOError, OSError):
        return default


class Sensors(threading.Thread):
    """Samples board sensors every `interval` seconds into a CSV; keeps power samples for energy estimates."""

    def __init__(self, path, interval=1.0):
        threading.Thread.__init__(self)
        self.daemon = True
        self.interval, self.stop_event = interval, threading.Event()
        self.zones = []
        for z in sorted(glob.glob('/sys/devices/virtual/thermal/thermal_zone*')):
            name = _read(z + '/type')
            if name and _read(z + '/temp') is not None:
                self.zones.append((name.replace(' ', '_'), z + '/temp'))
        gpu = ['/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq', '/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq']
        gpu += glob.glob('/sys/devices/*/devfreq/*gpu*/cur_freq') + glob.glob('/sys/class/devfreq/*gpu*/cur_freq')
        self.gpu = next((p for p in gpu if _read(p) is not None), None)
        self.cpu = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'
        self.rails = []                                   # (name, kind, paths): mW directly, or volts x amps
        for d in sorted(glob.glob('/sys/bus/i2c/drivers/ina3221x/*/iio:device*')):
            for k in range(3):
                p = '%s/in_power%d_input' % (d, k)
                if _read(p) is not None:
                    self.rails.append((_read('%s/rail_name_%d' % (d, k), 'rail%d' % k), 'mw', (p,)))
        for d in sorted(glob.glob('/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*')):
            for k in range(1, 4):
                v, i = '%s/in%d_input' % (d, k), '%s/curr%d_input' % (d, k)
                if _read(v) is not None and _read(i) is not None:
                    self.rails.append((_read('%s/in%d_label' % (d, k), 'rail%d' % k), 'vi', (v, i)))
        names = [r[0] for r in self.rails]
        self.total = next((n for n in names if n.upper() in ('POM_5V_IN', 'VDD_IN')), None)
        self.samples = []                                 # (t, total power W or None, max temp C)
        self.f = open(path, 'w')
        self.w = csv.writer(self.f)
        self.w.writerow(['t', 'ram_used_mb', 'cpu_mhz', 'gpu_mhz'] + ['temp_' + n for n, _ in self.zones] +
                        ['power_mw_' + n for n in names])

    def has_power(self):
        return self.total is not None

    def sample(self):
        mem = {}
        for line in (_read('/proc/meminfo', '') or '').splitlines():
            parts = line.split()
            if len(parts) >= 2:
                mem[parts[0].rstrip(':')] = float(parts[1]) / 1024.0
        ram = mem.get('MemTotal', 0) - mem.get('MemAvailable', 0)
        cpu = float(_read(self.cpu, 'nan')) / 1e3
        gpu = float(_read(self.gpu, 'nan')) / 1e6 if self.gpu else float('nan')
        temps = [float(_read(p, 'nan')) / 1e3 for _, p in self.zones]
        powers = []
        for name, kind, paths in self.rails:
            if kind == 'mw':
                powers.append(float(_read(paths[0], 'nan')))
            else:
                powers.append(float(_read(paths[0], 'nan')) * float(_read(paths[1], 'nan')) / 1e3)
        t = time.time()
        total = None
        if self.total is not None:
            total = powers[[r[0] for r in self.rails].index(self.total)] / 1e3
        good = [x for x in temps if x == x]
        self.samples.append((t, total, max(good) if good else float('nan')))
        self.w.writerow(['%.2f' % t, '%.0f' % ram, '%.0f' % cpu, '%.0f' % gpu] + ['%.1f' % x for x in temps] +
                        ['%.0f' % x for x in powers])

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.sample()
                self.f.flush()
            except Exception:                              # a sensor read must never stop the benchmark
                pass
            self.stop_event.wait(self.interval)

    def window(self, t0, t1):
        """Mean power (W, or None) and max temperature (C) between two times."""
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return None, float('nan')
        p = [x[1] for x in s if x[1] is not None]
        temps = [x[2] for x in s if x[2] == x[2]]
        return (float(np.mean(p)) if p else None), (max(temps) if temps else float('nan'))

    def close(self):
        self.stop_event.set()
        self.join(timeout=5)
        self.f.close()


def start_tegrastats(path):
    try:
        return subprocess.Popen(['tegrastats', '--interval', '1000', '--logfile', path],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ staged caption
def staged_caption(M, px, mode, max_new):
    """Same computation as SmolEdgeVLMJetson.caption, with a synchronised timer after every stage."""
    st = OrderedDict()
    sync, now = M.sync, time.perf_counter
    k, L, eps = M.k, M.v['layers'], M.v['ln_eps']
    with torch.no_grad():
        sync(); t = now()
        def lap(name):
            sync(); t1 = now(); st[name] = (t1 - lap.t) * 1e3; lap.t = t1
        lap.t = t
        h = M.embed(px); lap('embed')
        for i in range(k):
            h = M.vit_layer(i, h)
        lap('layers_1_3')
        if mode == 'full':
            for i in range(k, L):
                h = M.vit_layer(i, h)
            lap('layers_4_12')
            feats = M.ln(h, 'v.post', eps); lap('post_norm')
        else:
            m, sizes, assign = M.merge(h); lap('merge')
            m = M.restore(m); lap('restore')
            bias = torch.log(sizes)[None, None, None, :].to(M.dtype)
            for i in range(k, L):
                m = M.vit_layer(i, m, bias)
            lap('layers_4_12')
            feats = M.ln(m, 'v.post', eps)[:, assign, :]; lap('post_norm')
        Z = M.connector(feats, mode); lap('connector')
        seq = torch.cat([Z, M.prompt], 1)
        cache = [None] * M.l['layers']
        logits, cache = M.lm_step(seq, cache, 0)
        nxt = int(logits.argmax(-1)); lap('prefill')
        ttft = (lap.t - t) * 1e3
        toks, steps, pos = [], [], seq.shape[1]
        for _ in range(max_new):
            if nxt == M.eos:
                break
            toks.append(nxt)
            e = M.W['lm.embed'][nxt][None, None]
            logits, cache = M.lm_step(e, cache, pos)
            pos += 1
            nxt = int(logits.argmax(-1))
            sync(); t1 = now(); steps.append((t1 - lap.t) * 1e3); lap.t = t1
        total = (lap.t - t) * 1e3
    return st, steps, ttft, total, M.decode(toks), len(toks)


# ------------------------------------------------------------------ helpers
def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(np.ceil(q / 100.0 * len(xs))) - 1)] if xs else float('nan')


def med(xs):
    return float(np.median(xs)) if len(xs) else float('nan')


def append_rows(path, header, rows):
    new = not os.path.exists(path)
    with open(path, 'a') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def banner(s):
    print('\n' + '=' * 100 + '\n' + s + '\n' + '=' * 100, flush=True)


PER_IMAGE = ['image', 'phase', 'mode', 'caption', 'tokens', 'ttft_ms', 'total_ms', 'runs', 'matches_reference', 't_start', 't_end']
STAGES = ['image', 'mode', 'embed', 'layers_1_3', 'merge', 'restore', 'layers_4_12', 'post_norm', 'connector', 'prefill',
          'ttft_staged', 'decode_ms_per_token', 'tokens', 'total_staged']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=os.path.join(HERE, 'smoledgevlm_jetson_weights.npz'))
    ap.add_argument('--meta', default=os.path.join(HERE, 'smoledgevlm_jetson_meta.json'))
    default_images = os.path.join(HERE, 'images_test500')
    ap.add_argument('--images', default=default_images if os.path.isdir(default_images) else os.path.join(HERE, 'images'))
    ap.add_argument('--limit', type=int, default=0, help='use only the first N images (0 = all)')
    ap.add_argument('--timing-images', type=int, default=100, help='images that get repeated runs and stage timing')
    ap.add_argument('--runs', type=int, default=3, help='timed runs per pipeline for the timing images')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--max-new', type=int, default=40, help='caption length cap (40 = same as the T4 comparison)')
    ap.add_argument('--sustain-min', type=float, default=5.0, help='minutes of continuous captioning per pipeline (0 = skip)')
    ap.add_argument('--idle-s', type=float, default=30.0)
    ap.add_argument('--dtype', default='float16', choices=['float16', 'float32'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=os.path.join(HERE, 'results'))
    ap.add_argument('--resume', action='store_true', help='skip images already in per_image.csv')
    ap.add_argument('--quick', action='store_true', help='short check: 6 images, 1 run, 0.5 min sustained')
    a = ap.parse_args()
    if a.quick:
        a.limit, a.timing_images, a.runs, a.warmup, a.sustain_min, a.idle_s = 6, 3, 1, 1, 0.5, 5.0
        a.out = a.out + '_quick'
    os.makedirs(a.out, exist_ok=True)
    if not a.resume:
        for n in ('per_image.csv', 'stages.csv', 'sustained.csv'):
            p = os.path.join(a.out, n)
            if os.path.exists(p):
                os.rename(p, p + '.old')
    dtype = torch.float16 if a.dtype == 'float16' else torch.float32
    files = sorted(f for f in glob.glob(os.path.join(a.images, '*')) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit('no images found in %s' % a.images)
    T = min(a.timing_images, len(files))
    est_h = (T * (a.runs + 1) * 13.0 + (len(files) - T) * 13.0 + 2 * a.sustain_min * 60 + a.idle_s + 120) / 3600.0
    pmode = _read('/var/lib/nvpmodel/status', 'unknown')
    banner('SmolEdgeVLM Jetson benchmark | torch %s | %s | %s | %d images (%d with stage timing) | max %d tokens\n'
           'power mode: %s | rough time at 5 W: %.1f h | results -> %s' %
           (torch.__version__, a.device, a.dtype, len(files), T, a.max_new, pmode, est_h, a.out))

    sens = Sensors(os.path.join(a.out, 'sensors.csv'))
    sens.start()
    teg = start_tegrastats(os.path.join(a.out, 'tegrastats.log'))
    print('sensors: %d temperature zones, GPU clock %s, power rails: %s | tegrastats: %s' % (
        len(sens.zones), 'yes' if sens.gpu else 'no', ', '.join(r[0] for r in sens.rails) or 'none (use a USB power meter)',
        'logging' if teg else 'not available'), flush=True)

    t = time.time()
    M = jf.SmolEdgeVLMJetson(a.weights, a.meta, a.device, dtype)
    peak, avail = jf.mem_mb()
    print('model loaded in %.1f s | process peak %.0f MB | available now %.0f MB' % (time.time() - t, peak, avail), flush=True)
    ref = M.meta.get('reference', {})
    modes = ['full', 'smoledgevlm']
    px0 = M.pixels(Image.open(files[0]))
    for m in modes:
        for _ in range(a.warmup):
            M.caption(px0, m, a.max_new)
    print('warm-up done', flush=True)

    per_path, st_path = os.path.join(a.out, 'per_image.csv'), os.path.join(a.out, 'stages.csv')
    done = set()
    if a.resume:
        cnt = {}
        for r in read_rows(per_path):
            cnt[r['image']] = cnt.get(r['image'], 0) + 1
        done = set(k for k, v in cnt.items() if v >= 2)
        print('resume: %d images already done' % len(done), flush=True)

    # ---------------------------------------------------------- phases 1 and 2
    t_run = time.time()
    n_new = 0
    for n, f in enumerate(files):
        name = os.path.basename(f)
        if name in done:
            continue
        timing = n < T
        px = M.pixels(Image.open(f))
        runs = a.runs if timing else 1
        res = dict((m, []) for m in modes)
        t0 = time.time()
        for _ in range(runs):
            for m in modes:
                res[m].append(M.caption(px, m, a.max_new))
        t1 = time.time()
        rows = []
        for m in modes:
            cap = res[m][0][0]
            ntok = res[m][0][3]
            r_ = ref.get(name, {}).get(m)                 # references were made with a 24-token cap
            match = '' if r_ is None else (cap == r_ or (ntok >= 24 and cap.startswith(r_)))
            rows.append([name, 'timing' if timing else 'captions', m, cap, res[m][0][3],
                         '%.1f' % med([r[1] for r in res[m]]), '%.1f' % med([r[2] for r in res[m]]), runs, match,
                         '%.2f' % t0, '%.2f' % t1])
        append_rows(per_path, PER_IMAGE, rows)
        if timing:
            srows = []
            for m in modes:
                st, steps, ttft, total, cap, ntok = staged_caption(M, px, m, a.max_new)
                srows.append([name, m] + ['%.2f' % st[k] if k in st else '' for k in STAGES[2:10]] +
                             ['%.1f' % ttft, '%.1f' % med(steps), ntok, '%.1f' % total])
            append_rows(st_path, STAGES, srows)
        n_new += 1
        el = time.time() - t_run
        left = len(files) - n - 1
        per = el / n_new
        print('[%d/%d] %s | full %s ms | smoledgevlm %s ms | %s | about %.0f min left' % (
            n + 1, len(files), name, rows[0][5], rows[1][5], rows[1][3][:60], left * per / 60.0), flush=True)

    # ---------------------------------------------------------- phases 3 and 4
    sus_path = os.path.join(a.out, 'sustained.csv')
    idle = None
    if a.sustain_min > 0:
        banner('IDLE for %.0f s (baseline). If you use a USB power meter, note its watts now.' % a.idle_s)
        ti = time.time()
        time.sleep(a.idle_s)
        idle = sens.window(ti, time.time())
        sus = {}
        imgs = [M.pixels(Image.open(f)) for f in files[:min(len(files), 20)]]      # 20 images keep GPU memory small
        for m in ['smoledgevlm', 'full']:
            banner('SUSTAINED: %s only, %.1f min of continuous captioning. Note the power meter watts during this phase.' %
                   (m, a.sustain_min))
            ts, k_, rows = time.time(), 0, []
            while time.time() - ts < a.sustain_min * 60:
                c0 = time.time()
                cap, ttft, total, ntok = M.caption(imgs[k_ % len(imgs)], m, a.max_new)
                c1 = time.time()
                rows.append([m, k_, '%.2f' % c0, '%.2f' % c1, '%.1f' % ttft, '%.1f' % total, ntok])
                k_ += 1
            te = time.time()
            append_rows(sus_path, ['mode', 'i', 't_start', 't_end', 'ttft_ms', 'total_ms', 'tokens'], rows)
            p, tmax = sens.window(ts, te)
            first, last = rows[:max(1, len(rows) // 5)], rows[-max(1, len(rows) // 5):]
            sus[m] = {'captions': len(rows), 'minutes': (te - ts) / 60.0, 'captions_per_min': len(rows) / ((te - ts) / 60.0),
                      'ttft_ms_median': med([float(r[4]) for r in rows]), 'total_ms_median': med([float(r[5]) for r in rows]),
                      'ttft_ms_first_fifth': med([float(r[4]) for r in first]), 'ttft_ms_last_fifth': med([float(r[4]) for r in last]),
                      'power_w_mean': p, 'temp_c_max': tmax,
                      'energy_j_per_caption': (p * (te - ts) / len(rows)) if p is not None else None}
            print('%s: %d captions in %.1f min (%.1f per min) | TTFT median %.0f ms (first fifth %.0f, last fifth %.0f) | '
                  'max temp %s | power %s' % (m, len(rows), sus[m]['minutes'], sus[m]['captions_per_min'], sus[m]['ttft_ms_median'],
                                                 sus[m]['ttft_ms_first_fifth'], sus[m]['ttft_ms_last_fifth'], ('%.1f C' % tmax) if tmax == tmax else 'n/a',
                                                 ('%.2f W' % p) if p is not None else 'not reported by the board'), flush=True)
    else:
        sus = {}
    sens.close()
    if teg is not None:
        teg.terminate()

    # ---------------------------------------------------------- summary
    rows = read_rows(per_path)
    srows = read_rows(st_path)
    summ = {'device': a.device, 'dtype': a.dtype, 'torch': torch.__version__, 'power_mode': pmode, 'max_new': a.max_new,
            'images': len(set(r['image'] for r in rows)), 'timing_images': len(set(r['image'] for r in rows if r['phase'] == 'timing')),
            'runs': a.runs, 'process_peak_mb': jf.mem_mb()[0], 'modes': {}, 'stages_ms_median': {}, 'sustained': sus,
            'idle_power_w': idle[0] if idle else None, 'idle_temp_c_max': idle[1] if idle else None}
    for m in modes:
        tm = [r for r in rows if r['mode'] == m and r['phase'] == 'timing']
        allm = [r for r in rows if r['mode'] == m]
        refm = [r for r in allm if r['matches_reference'] in ('True', 'False')]
        summ['modes'][m] = {'ttft_median_ms': med([float(r['ttft_ms']) for r in tm]), 'ttft_p95_ms': pct([float(r['ttft_ms']) for r in tm], 95),
                            'total_median_ms': med([float(r['total_ms']) for r in tm]),
                            'tokens_mean': float(np.mean([int(r['tokens']) for r in allm])) if allm else float('nan'),
                            'captions': len(allm), 'reference_match': '%d/%d' % (sum(r['matches_reference'] == 'True' for r in refm), len(refm))}
        sm = [r for r in srows if r['mode'] == m]
        summ['stages_ms_median'][m] = OrderedDict((k, med([float(r[k]) for r in sm if r[k] != ''])) for k in STAGES[2:12]
                                                  if any(r[k] != '' for r in sm))
    f_, g_ = summ['modes']['full'], summ['modes']['smoledgevlm']
    if f_['ttft_median_ms'] == f_['ttft_median_ms']:
        summ['ttft_change_pct'] = 100.0 * (g_['ttft_median_ms'] / f_['ttft_median_ms'] - 1)
        summ['total_change_pct'] = 100.0 * (g_['total_median_ms'] / f_['total_median_ms'] - 1)
    json.dump(summ, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)

    banner('JETSON RESULT (%s, %s, power mode %s, %d images, stage timing on %d)' % (
        a.device, a.dtype, pmode, summ['images'], summ['timing_images']))
    print('%-11s %10s %9s %12s %13s %10s' % ('pipeline', 'TTFT ms', 'p95', 'caption ms', 'tokens (avg)', 'ref match'))
    for m in modes:
        s = summ['modes'][m]
        print('%-11s %10.1f %9.1f %12.1f %13.1f %10s' % (m, s['ttft_median_ms'], s['ttft_p95_ms'], s['total_median_ms'], s['tokens_mean'],
                                                        s['reference_match']))
    if 'ttft_change_pct' in summ:
        print('SmolEdgeVLM vs full: time to first token %+.1f%%, whole caption %+.1f%%' % (summ['ttft_change_pct'], summ['total_change_pct']))
    print('\nStage times, median ms (one synchronised pass per image):')
    keys = [k for k in STAGES[2:12] if k in summ['stages_ms_median']['smoledgevlm'] or k in summ['stages_ms_median']['full']]
    print('%-22s %10s %10s' % ('stage', 'full', 'smoledgevlm'))
    for k in keys:
        fv = summ['stages_ms_median']['full'].get(k, float('nan'))
        gv = summ['stages_ms_median']['smoledgevlm'].get(k, float('nan'))
        print('%-22s %10s %10s' % (k, '-' if fv != fv else '%.1f' % fv, '-' if gv != gv else '%.1f' % gv))
    if sus:
        print('\nSustained run (per pipeline):')
        for m in ('smoledgevlm', 'full'):
            s = sus[m]
            e = s['energy_j_per_caption']
            print('  %-10s %.1f captions/min | TTFT %.0f ms (first fifth %.0f, last fifth %.0f) | max temp %s | %s' % (
                m, s['captions_per_min'], s['ttft_ms_median'], s['ttft_ms_first_fifth'], s['ttft_ms_last_fifth'],
                ('%.1f C' % s['temp_c_max']) if s['temp_c_max'] == s['temp_c_max'] else 'n/a',
                ('%.2f W, %.1f J per caption' % (s['power_w_mean'], e)) if e is not None else 'power: read it from a USB meter'))
    print('\nwritten: %s' % ', '.join(sorted(os.listdir(a.out))), flush=True)


if __name__ == '__main__':
    main()
