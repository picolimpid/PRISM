#!/usr/bin/env python3
"""
Build positive/weak-negative text embeddings from user-feed-post samples.

Outputs:
- cleaned_examples.csv: optional cleaned text sample for inspection.
- user_label_embeddings.csv: one positive and one weak-negative mean embedding per user.
- global_label_embeddings.json: dataset-level positive and weak-negative mean embeddings.
- embedding_config.json: run metadata.

The embedding step requires sentence-transformers + torch in a Python environment that
can load the selected model. Default baseline model: BAAI/bge-m3.
"""

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

URL_RE = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)
WS_RE = re.compile(r'\s+')
DEFAULT_DATASET = Path('/home/hanqim2/data_pipeline/first2000_user_feed_post_dataset/user_feed_post_interactions.csv')
DEFAULT_OUTPUT_DIR = Path('/home/hanqim2/data_pipeline/first2000_user_feed_post_dataset/embeddings_bge_m3')


def drop_emoji_and_controls(text):
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ('Cc', 'Cf', 'Cs', 'Co', 'Cn'):
            if ch in ('\n', '\t'):
                chars.append(' ')
            continue
        # Most emoji and decorative icons are Symbol/Other. Keep currency/math symbols out too
        # for a conservative text-only baseline.
        if cat == 'So':
            continue
        chars.append(ch)
    return ''.join(chars)


def clean_text(text, keep_hashtags=True):
    if text is None:
        return ''
    text = str(text)
    text = unicodedata.normalize('NFKC', text)
    text = URL_RE.sub(' ', text)
    text = drop_emoji_and_controls(text)

    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
        elif keep_hashtags and ch in ('#', '@'):
            cleaned.append(ch)
        elif cat.startswith('P'):
            cleaned.append(' ')
        elif cat.startswith('S'):
            cleaned.append(' ')
        else:
            cleaned.append(ch)
    text = ''.join(cleaned)
    text = WS_RE.sub(' ', text).strip()
    return text


def compose_text(row, include_feed_name=False, include_comment_text=False):
    parts = []
    if include_feed_name and row.get('feed'):
        parts.append('feed ' + row.get('feed', ''))
    if row.get('post_text'):
        parts.append(row.get('post_text', ''))
    if include_comment_text and row.get('comment_content_sample'):
        parts.append(row.get('comment_content_sample', ''))
    return ' '.join(parts)


def iter_rows(dataset_path, max_rows=None):
    with dataset_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            yield idx, row


def inspect_cleaning(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / 'cleaned_examples.csv'
    written = 0
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        fieldnames = ['row_index', 'user_id', 'feed', 'post_id', 'label', 'raw_text', 'cleaned_text']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in iter_rows(args.dataset, args.max_rows):
            raw = compose_text(row, args.include_feed_name, args.include_comment_text)
            cleaned = clean_text(raw, keep_hashtags=args.keep_hashtags)
            if not cleaned:
                continue
            writer.writerow({
                'row_index': idx,
                'user_id': row.get('user_id'),
                'feed': row.get('feed'),
                'post_id': row.get('post_id'),
                'label': row.get('label'),
                'raw_text': raw,
                'cleaned_text': cleaned,
            })
            written += 1
            if args.cleaned_examples and written >= args.cleaned_examples:
                break
    print(json.dumps({'cleaned_examples': str(output_path), 'rows_written': written}, indent=2))


def vector_to_json(vector, precision):
    return json.dumps([round(float(x), precision) for x in vector], separators=(',', ':'))


def add_sum(target, vector):
    if target is None:
        return vector.astype('float64')
    target += vector
    return target


def run_embedding(args):
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            'Missing embedding dependencies: {}\n'
            'Install in a newer Python environment, for example:\n'
            '  pip install torch sentence-transformers\n'
            'Then rerun this script.'.format(exc)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model_name, device=args.device)

    user_sums = defaultdict(lambda: {'positive': None, 'weak_negative': None})
    user_counts = defaultdict(lambda: {'positive': 0, 'weak_negative': 0})
    global_sums = {'positive': None, 'weak_negative': None}
    global_counts = {'positive': 0, 'weak_negative': 0}

    batch_texts = []
    batch_meta = []
    processed_rows = 0
    skipped_rows = 0

    def flush_batch():
        if not batch_texts:
            return
        embeddings = model.encode(
            batch_texts,
            batch_size=args.encode_batch_size,
            normalize_embeddings=args.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=args.show_progress,
        )
        for emb, meta in zip(embeddings, batch_meta):
            user_id, label = meta
            emb64 = emb.astype('float64')
            user_sums[user_id][label] = add_sum(user_sums[user_id][label], emb64)
            user_counts[user_id][label] += 1
            global_sums[label] = add_sum(global_sums[label], emb64)
            global_counts[label] += 1
        batch_texts[:] = []
        batch_meta[:] = []

    for idx, row in iter_rows(args.dataset, args.max_rows):
        label = row.get('label')
        if label not in ('positive', 'weak_negative'):
            skipped_rows += 1
            continue
        raw = compose_text(row, args.include_feed_name, args.include_comment_text)
        cleaned = clean_text(raw, keep_hashtags=args.keep_hashtags)
        if not cleaned:
            skipped_rows += 1
            continue
        batch_texts.append(cleaned)
        batch_meta.append((row.get('user_id'), label))
        processed_rows += 1
        if len(batch_texts) >= args.text_batch_size:
            flush_batch()
    flush_batch()

    user_output = args.output_dir / 'user_label_embeddings.csv'
    with user_output.open('w', encoding='utf-8', newline='') as handle:
        fieldnames = ['user_id', 'positive_count', 'weak_negative_count', 'positive_embedding', 'weak_negative_embedding']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for user_id in sorted(user_counts, key=lambda x: int(x) if str(x).isdigit() else str(x)):
            row = {'user_id': user_id}
            for label in ('positive', 'weak_negative'):
                count = user_counts[user_id][label]
                row[label + '_count'] = count
                if count > 0:
                    mean_vec = user_sums[user_id][label] / float(count)
                    if args.normalize_profile_embeddings:
                        norm = np.linalg.norm(mean_vec)
                        if norm > 0:
                            mean_vec = mean_vec / norm
                    row[label + '_embedding'] = vector_to_json(mean_vec, args.precision)
                else:
                    row[label + '_embedding'] = ''
            writer.writerow(row)

    global_output = args.output_dir / 'global_label_embeddings.json'
    global_payload = {
        'model_name': args.model_name,
        'processed_rows': processed_rows,
        'skipped_rows': skipped_rows,
        'counts': global_counts,
        'embeddings': {},
    }
    for label in ('positive', 'weak_negative'):
        count = global_counts[label]
        if count > 0:
            mean_vec = global_sums[label] / float(count)
            if args.normalize_profile_embeddings:
                norm = np.linalg.norm(mean_vec)
                if norm > 0:
                    mean_vec = mean_vec / norm
            global_payload['embeddings'][label] = [round(float(x), args.precision) for x in mean_vec]
        else:
            global_payload['embeddings'][label] = []
    with global_output.open('w', encoding='utf-8') as handle:
        json.dump(global_payload, handle, indent=2, sort_keys=True)

    config_output = args.output_dir / 'embedding_config.json'
    config = vars(args).copy()
    config['dataset'] = str(args.dataset)
    config['output_dir'] = str(args.output_dir)
    with config_output.open('w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2, sort_keys=True, default=str)

    print(json.dumps({
        'output_dir': str(args.output_dir),
        'user_embeddings': str(user_output),
        'global_embeddings': str(global_output),
        'processed_rows': processed_rows,
        'skipped_rows': skipped_rows,
        'positive_count': global_counts['positive'],
        'weak_negative_count': global_counts['weak_negative'],
    }, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(description='Clean text and build positive/weak-negative profile embeddings.')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--model-name', default='BAAI/bge-m3')
    parser.add_argument('--device', default=None, help='Example: cuda, cuda:0, cpu. Default lets sentence-transformers choose.')
    parser.add_argument('--text-batch-size', type=int, default=512)
    parser.add_argument('--encode-batch-size', type=int, default=16)
    parser.add_argument('--precision', type=int, default=6)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--include-feed-name', action='store_true')
    parser.add_argument('--include-comment-text', action='store_true')
    parser.add_argument('--keep-hashtags', action='store_true', default=True)
    parser.add_argument('--drop-hashtags', dest='keep_hashtags', action='store_false')
    parser.add_argument('--normalize-embeddings', action='store_true', default=True)
    parser.add_argument('--no-normalize-embeddings', dest='normalize_embeddings', action='store_false')
    parser.add_argument('--normalize-profile-embeddings', action='store_true', default=True)
    parser.add_argument('--no-normalize-profile-embeddings', dest='normalize_profile_embeddings', action='store_false')
    parser.add_argument('--show-progress', action='store_true')
    parser.add_argument('--clean-only', action='store_true')
    parser.add_argument('--cleaned-examples', type=int, default=200)
    return parser


def main():
    args = build_parser().parse_args()
    if args.clean_only:
        inspect_cleaning(args)
    else:
        run_embedding(args)


if __name__ == '__main__':
    main()
