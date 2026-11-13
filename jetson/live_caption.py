#!/usr/bin/env python3
"""Live captioning on the Jetson Nano with the SmolEdgeVLM board code.

Captions frames from a USB or CSI camera (or a folder of photos), shows timing, and saves every frame with its
caption so the same scenes can be given to other models later. Optionally asks you to rate each caption.

    python3 live_caption.py --weights W.npz --meta M.json --camera usb --rate
    python3 live_caption.py --weights W.npz --meta M.json --camera csi --every 5
    python3 live_caption.py --weights W.npz --meta M.json --images my_photos --rate

Needs smoledgevlm_jetson.py in the same folder. OpenCV is needed only for a camera (JetPack ships it).
Output folder: frames/*.jpg, live_log.csv (one row per caption) and live_summary.json.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from smoledgevlm_jetson import SmolEdgeVLMJetson, mem_mb  # noqa: E402

RATINGS = {'c': 'correct', 'p': 'partly correct', 'w': 'wrong'}


def board_temperature_c():
    """Highest reading over the board's thermal zones, in degrees Celsius (nan if unavailable)."""
    highest = float('nan')
    for path in glob.glob('/sys/devices/virtual/thermal/thermal_zone*/temp'):
        try:
            value = int(open(path).read().strip()) / 1000.0
        except (IOError, ValueError):
            continue
        if np.isnan(highest) or value > highest:
            highest = value
    return highest


def open_camera(kind, index, width, height):
    import cv2
    if kind == 'csi':
        pipeline = ('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, framerate=30/1 ! '
                    'nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1'
                    % (width, height))
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit('could not open the %s camera' % kind)
    return cap


def grab_frame(cap):
    """Read a few frames so the image is current (USB cameras buffer old frames), return a PIL image."""
    import cv2
    frame = None
    for _ in range(4):
        ok, frame = cap.read()
        if not ok:
            frame = None
    if frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def ask_rating():
    print('  rate: [c]orrect  [p]artly  [w]rong, add h if it names something not there (e.g. "wh"), Enter to skip,'
          ' q to quit')
    answer = sys.stdin.readline().strip().lower()
    if answer == 'q':
        return None, None, True
    quality = ''
    hallucination = ''
    if answer:
        quality = RATINGS.get(answer[0], '')
        if 'h' in answer:
            hallucination = 'yes'
        else:
            hallucination = 'no'
    return quality, hallucination, False


def highest_temperature(rows):
    highest = float('nan')
    for row in rows:
        value = row['temp_c']
        if np.isnan(value):
            continue
        if np.isnan(highest) or value > highest:
            highest = value
    return highest


def summarise(rows):
    ttft = []
    total = []
    for row in rows:
        ttft.append(row['ttft_ms'])
        total.append(row['caption_ms'])
    rated = []
    for row in rows:
        if row['rating'] != '':
            rated.append(row)
    counts = {'correct': 0, 'partly correct': 0, 'wrong': 0}
    hallucinated = 0
    for row in rated:
        counts[row['rating']] += 1
        if row['hallucination'] == 'yes':
            hallucinated += 1
    summary = {'captions': len(rows),
               'ttft_median_ms': float(np.median(ttft)) if ttft else float('nan'),
               'caption_median_ms': float(np.median(total)) if total else float('nan'),
               'caption_p95_ms': float(np.percentile(total, 95)) if total else float('nan'),
               'rated': len(rated), 'ratings': counts, 'hallucinated': hallucinated,
               'process_peak_mb': mem_mb()[0], 'max_temperature_c': highest_temperature(rows)}
    if rated:
        summary['useful_share'] = (counts['correct'] + counts['partly correct']) / float(len(rated))
        summary['correct_share'] = counts['correct'] / float(len(rated))
        summary['hallucination_share'] = hallucinated / float(len(rated))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--camera', default='usb', choices=['usb', 'csi', 'none'])
    ap.add_argument('--cam-index', type=int, default=0)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--images', default='', help='caption a folder of photos instead of a camera')
    ap.add_argument('--every', type=float, default=0, help='seconds between captions (0 = press Enter each time)')
    ap.add_argument('--count', type=int, default=0, help='stop after this many captions (0 = until q or Ctrl+C)')
    ap.add_argument('--rate', action='store_true', help='ask for a rating after each caption')
    ap.add_argument('--max-new', type=int, default=40)
    ap.add_argument('--mode', default='smoledgevlm', choices=['smoledgevlm', 'full'])
    ap.add_argument('--out', default=os.path.join(HERE, 'live_results'))
    a = ap.parse_args()

    frame_dir = os.path.join(a.out, 'frames')
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir)

    print('loading model ...')
    started = time.time()
    model = SmolEdgeVLMJetson(a.weights, a.meta, 'cuda', torch.float16)
    print('loaded in %.1f s | process peak %.0f MB | available %.0f MB' % ((time.time() - started,) + mem_mb()))

    files = []
    cap = None
    if a.images:
        for path in sorted(glob.glob(os.path.join(a.images, '*'))):
            if path.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(path)
        if not files:
            raise SystemExit('no .jpg/.png files in %s' % a.images)
    elif a.camera != 'none':
        cap = open_camera(a.camera, a.cam_index, a.width, a.height)
    else:
        raise SystemExit('give --images or a camera')

    warm = Image.new('RGB', (512, 512), (128, 128, 128))
    warm_px = model.pixels(warm)
    for _ in range(3):
        model.caption(warm_px, a.mode, a.max_new)
    print('warm-up done\n')

    rows = []
    n = 0
    try:
        while True:
            if a.count and n >= a.count:
                break
            if files:
                if n >= len(files):
                    break
                image = Image.open(files[n]).convert('RGB')
                source = os.path.basename(files[n])
            else:
                if a.every > 0:
                    if n > 0:
                        time.sleep(a.every)
                else:
                    print('point the camera and press Enter (q + Enter to quit)')
                    if sys.stdin.readline().strip().lower() == 'q':
                        break
                image = grab_frame(cap)
                if image is None:
                    print('camera returned no frame, retrying')
                    continue
                source = 'camera'
            n += 1
            t0 = time.perf_counter()
            px = model.pixels(image)
            prep_ms = (time.perf_counter() - t0) * 1e3
            caption, ttft_ms, caption_ms, tokens = model.caption(px, a.mode, a.max_new)
            temp_c = board_temperature_c()
            name = 'frame_%04d.jpg' % n
            image.save(os.path.join(frame_dir, name), quality=92)
            print('[%d] %s' % (n, caption))
            print('    first word %.0f ms | caption %.0f ms | %d tokens | preprocessing %.0f ms | %.1f C'
                  % (ttft_ms, caption_ms, tokens, prep_ms, temp_c))
            quality = ''
            hallucination = ''
            if a.rate:
                quality, hallucination, stop = ask_rating()
                if stop:
                    break
            rows.append({'frame': name, 'source': source, 'caption': caption, 'ttft_ms': ttft_ms,
                         'caption_ms': caption_ms, 'tokens': tokens, 'prep_ms': prep_ms, 'temp_c': temp_c,
                         'rating': quality, 'hallucination': hallucination, 'note': ''})
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        if cap is not None:
            cap.release()

    if not rows:
        print('nothing captioned')
        return
    with open(os.path.join(a.out, 'live_log.csv'), 'w') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    summary = summarise(rows)
    json.dump(summary, open(os.path.join(a.out, 'live_summary.json'), 'w'), indent=1)
    print('\n' + '=' * 70)
    print('captions %d | first word median %.0f ms | caption median %.0f ms (p95 %.0f) | peak %.0f MB | max %.1f C'
          % (summary['captions'], summary['ttft_median_ms'], summary['caption_median_ms'], summary['caption_p95_ms'],
             summary['process_peak_mb'], summary['max_temperature_c']))
    if summary['rated']:
        print('rated %d | correct %.0f%% | correct or partly %.0f%% | hallucination %.0f%%'
              % (summary['rated'], 100 * summary['correct_share'], 100 * summary['useful_share'],
                 100 * summary['hallucination_share']))
    print('saved in %s' % a.out)


if __name__ == '__main__':
    main()
