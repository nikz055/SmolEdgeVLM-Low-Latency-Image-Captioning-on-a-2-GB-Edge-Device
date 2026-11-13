"""Score a Jetson board run against the compact models on the same 100 COCO images.

Uses the same scorer and fair rule as the board comparison: each caption is cut to its first sentence and stock
openings such as "In this image we can see" are removed, then BLEU-4, METEOR, ROUGE-L and CIDEr are computed against
the five COCO references. CIDEr differences come with 95% paired bootstrap intervals over the images.

    python scripts/score_board.py \
        --comparison results/jetson_board_100/captions_all_models.csv \
        --run results/v2/jetson_board_100/per_image.csv \
        --name "SmolEdgeVLM v2 + LoRA (16-bit)" \
        --out results/v2/jetson_board_100/scores.json

Needs pycocoevalcap (and Java for METEOR and the PTB tokenizer).
"""
import argparse
import csv
import json
import re

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

STOCK_OPENING = re.compile(
    r'^\s*(in (the|this) image,?|the image (shows|depicts|showcases|features)|this image (shows|depicts))'
    r'\s*(,\s*)?(we (can )?see|there (is|are)|i can see|we can see)?\s*',
    re.I,
)

BOOTSTRAP_SAMPLES = 10000


def fair_caption(text):
    """First sentence, without a stock opening, with a capital first letter."""
    sentence_match = re.match(r'(.+?[.!?])(\s|$)', text)
    if sentence_match:
        first_sentence = sentence_match.group(1)
    else:
        first_sentence = text
    trimmed = STOCK_OPENING.sub('', first_sentence).strip()
    if trimmed:
        return trimmed[:1].upper() + trimmed[1:]
    return first_sentence


def image_id_from_file(file_name):
    """COCO_val2014_000000004256.jpg -> '4256'."""
    number_part = file_name.split('_')[-1].split('.')[0]
    return str(int(number_part))


def read_comparison(path):
    """References and the captions of every model in the comparison table."""
    references = {}
    captions_by_model = {}
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        model_columns = []
        for column in reader.fieldnames:
            if column in ('image_id', 'file'):
                continue
            if column.startswith('reference_'):
                continue
            model_columns.append(column)
        for row in reader:
            image_id = row['image_id']
            references[image_id] = []
            for index in range(1, 6):
                references[image_id].append(row['reference_' + str(index)])
            for column in model_columns:
                if column not in captions_by_model:
                    captions_by_model[column] = {}
                captions_by_model[column][image_id] = row[column]
    return references, captions_by_model


def read_run(path, mode):
    """Captions of one pipeline from a jetson/benchmark.py per_image.csv."""
    captions = {}
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['mode'] != mode:
                continue
            captions[image_id_from_file(row['image'])] = row['caption']
    return captions


def score_model(tokenizer, tokenized_references, image_ids, captions):
    """Corpus scores and per-image CIDEr for one model."""
    raw_results = {}
    for image_id in image_ids:
        raw_results[image_id] = [{'caption': fair_caption(captions[image_id])}]
    tokenized_results = tokenizer.tokenize(raw_results)
    bleu_scores, _ = Bleu(4).compute_score(tokenized_references, tokenized_results, verbose=0)
    meteor_score, _ = Meteor().compute_score(tokenized_references, tokenized_results)
    rouge_score, _ = Rouge().compute_score(tokenized_references, tokenized_results)
    cider_score, cider_per_image = Cider().compute_score(tokenized_references, tokenized_results)
    scores = {
        'BLEU-4': round(float(bleu_scores[3]), 3),
        'METEOR': round(float(meteor_score), 3),
        'ROUGE-L': round(float(rouge_score), 3),
        'CIDEr': round(float(cider_score), 3),
    }
    return scores, np.array(cider_per_image)


def paired_bootstrap(differences, image_count):
    """Mean difference and its 95% interval, resampling images with replacement."""
    generator = np.random.default_rng(0)
    sample_indices = generator.integers(0, image_count, size=(BOOTSTRAP_SAMPLES, image_count))
    sample_means = differences[sample_indices].mean(axis=1)
    low = float(np.percentile(sample_means, 2.5))
    high = float(np.percentile(sample_means, 97.5))
    return {'mean': round(float(differences.mean()), 3), 'ci95': [round(low, 3), round(high, 3)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', required=True, help='captions_all_models.csv of the board comparison')
    parser.add_argument('--run', required=True, help='per_image.csv written by jetson/benchmark.py')
    parser.add_argument('--mode', default='smoledgevlm', help='pipeline to score from the run')
    parser.add_argument('--name', default='new run', help='name of the scored run in the output')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    references, captions_by_model = read_comparison(args.comparison)
    run_captions = read_run(args.run, args.mode)
    image_ids = sorted(references.keys(), key=int)
    missing = []
    for image_id in image_ids:
        if image_id not in run_captions:
            missing.append(image_id)
    if missing:
        raise SystemExit('the run has no caption for %d of the %d images' % (len(missing), len(image_ids)))

    tokenizer = PTBTokenizer()
    raw_references = {}
    for image_id in image_ids:
        raw_references[image_id] = []
        for reference in references[image_id]:
            raw_references[image_id].append({'caption': reference})
    tokenized_references = tokenizer.tokenize(raw_references)

    all_captions = {args.name: run_captions}
    for model_name, captions in captions_by_model.items():
        all_captions[model_name] = captions

    scores = {}
    per_image_cider = {}
    for model_name, captions in all_captions.items():
        model_scores, model_cider = score_model(tokenizer, tokenized_references, image_ids, captions)
        scores[model_name] = model_scores
        per_image_cider[model_name] = model_cider
        print('%-48s BLEU-4 %.3f  METEOR %.3f  ROUGE-L %.3f  CIDEr %.3f' % (
            model_name, model_scores['BLEU-4'], model_scores['METEOR'], model_scores['ROUGE-L'], model_scores['CIDEr']))

    bootstrap = {}
    for model_name in captions_by_model:
        differences = per_image_cider[args.name] - per_image_cider[model_name]
        bootstrap[model_name] = paired_bootstrap(differences, len(image_ids))
        interval = bootstrap[model_name]['ci95']
        print('%s minus %-40s CIDEr %+.3f [%+.3f, %+.3f]' % (
            args.name, model_name, bootstrap[model_name]['mean'], interval[0], interval[1]))

    output = {
        'images': len(image_ids),
        'rule': 'first sentence, stock openings removed, five COCO references',
        'scores': scores,
        'cider_difference_' + args.name.replace(' ', '_') + '_minus': bootstrap,
    }
    with open(args.out, 'w') as handle:
        json.dump(output, handle, indent=1)
    print('written', args.out)


if __name__ == '__main__':
    main()
