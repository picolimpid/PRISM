#!/usr/bin/env python3
"""
PCA-based analysis for user positive/weak-negative profile embeddings.

Reads user_label_embeddings.csv and analyzes:
- positive embeddings
- weak-negative embeddings
- preference direction: positive - weak_negative

Outputs PCA coordinates, explained variance, KMeans cluster summaries,
nearest-neighbor pairs, and scatter plots.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_INPUT = Path('/home/hanqim2/data_pipeline/first2000_user_feed_post_dataset/embeddings_bge_m3/user_label_embeddings.csv')
DEFAULT_OUTPUT_DIR = Path('/home/hanqim2/data_pipeline/first2000_user_feed_post_dataset/embeddings_bge_m3/pca_analysis')


def parse_embedding(value):
    if value is None or not str(value).strip():
        return None
    return np.asarray(json.loads(value), dtype=np.float32)


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_embeddings(path, min_positive_count, min_negative_count):
    rows = []
    positive = []
    negative = []
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pos_count = int(row.get('positive_count') or 0)
            neg_count = int(row.get('weak_negative_count') or 0)
            if pos_count < min_positive_count or neg_count < min_negative_count:
                continue
            pos = parse_embedding(row.get('positive_embedding'))
            neg = parse_embedding(row.get('weak_negative_embedding'))
            if pos is None or neg is None:
                continue
            rows.append({
                'user_id': row['user_id'],
                'positive_count': pos_count,
                'weak_negative_count': neg_count,
            })
            positive.append(pos)
            negative.append(neg)

    if not rows:
        raise RuntimeError('No users passed the count filters.')

    positive = np.vstack(positive)
    negative = np.vstack(negative)
    diff = positive - negative
    return rows, positive, negative, diff


def run_pca(matrix, n_components):
    n_components = min(n_components, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(matrix)
    return pca, coords


def choose_kmeans(coords, max_k, seed):
    max_k = min(max_k, coords.shape[0] - 1)
    if max_k < 2:
        return None, None, []
    scores = []
    best = None
    best_score = -1.0
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(coords[:, : min(10, coords.shape[1])])
        score = silhouette_score(coords[:, : min(10, coords.shape[1])], labels)
        scores.append({'k': k, 'silhouette': float(score)})
        if score > best_score:
            best_score = score
            best = labels
    return best, best_score, scores


def summarize_clusters(rows, coords, labels):
    summaries = []
    if labels is None:
        return summaries
    labels = np.asarray(labels)
    for cluster_id in sorted(set(labels.tolist())):
        idx = np.where(labels == cluster_id)[0]
        pos_counts = np.asarray([rows[i]['positive_count'] for i in idx], dtype=float)
        neg_counts = np.asarray([rows[i]['weak_negative_count'] for i in idx], dtype=float)
        pc1 = coords[idx, 0]
        pc2 = coords[idx, 1] if coords.shape[1] > 1 else np.zeros(len(idx))
        summaries.append({
            'cluster': int(cluster_id),
            'n_users': int(len(idx)),
            'mean_pc1': float(np.mean(pc1)),
            'mean_pc2': float(np.mean(pc2)),
            'mean_positive_count': float(np.mean(pos_counts)),
            'mean_weak_negative_count': float(np.mean(neg_counts)),
            'example_user_ids': '|'.join(rows[i]['user_id'] for i in idx[:10]),
        })
    return summaries


def nearest_pairs(rows, matrix, top_n):
    sims = cosine_similarity(l2_normalize(matrix))
    np.fill_diagonal(sims, -np.inf)
    pairs = []
    seen = set()
    flat_order = np.argsort(sims.ravel())[::-1]
    for flat in flat_order:
        i, j = np.unravel_index(flat, sims.shape)
        key = tuple(sorted((int(i), int(j))))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            'user_id_a': rows[i]['user_id'],
            'user_id_b': rows[j]['user_id'],
            'cosine_similarity': float(sims[i, j]),
            'a_positive_count': rows[i]['positive_count'],
            'b_positive_count': rows[j]['positive_count'],
            'a_weak_negative_count': rows[i]['weak_negative_count'],
            'b_weak_negative_count': rows[j]['weak_negative_count'],
        })
        if len(pairs) >= top_n:
            break
    return pairs


def write_csv(path, rows, fieldnames):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_scatter(path, coords, rows, labels, title, explained):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    sizes = np.asarray([rows[i]['positive_count'] for i in range(len(rows))], dtype=float)
    sizes = 12 + 45 * np.sqrt(sizes / max(sizes.max(), 1.0))
    color_values = labels if labels is not None else np.asarray([rows[i]['weak_negative_count'] for i in range(len(rows))])
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=color_values, s=sizes, alpha=0.78, cmap='tab10' if labels is not None else 'viridis', linewidths=0)
    ax.set_title(title)
    ax.set_xlabel('PC1 ({:.1f}% var)'.format(explained[0] * 100.0))
    ax.set_ylabel('PC2 ({:.1f}% var)'.format(explained[1] * 100.0 if len(explained) > 1 else 0.0))
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('cluster' if labels is not None else 'weak negative count')
    fig.tight_layout()
    fig.savefig(str(path))
    plt.close(fig)


def analyze_space(name, matrix, rows, output_dir, n_components, max_clusters, seed, nearest_top_n):
    matrix = l2_normalize(matrix)
    pca, coords = run_pca(matrix, n_components)
    labels, best_score, k_scores = choose_kmeans(coords, max_clusters, seed)

    coord_rows = []
    for i, row in enumerate(rows):
        out = dict(row)
        for j in range(coords.shape[1]):
            out['pc{}'.format(j + 1)] = float(coords[i, j])
        out['cluster'] = int(labels[i]) if labels is not None else ''
        coord_rows.append(out)

    coord_fields = ['user_id', 'positive_count', 'weak_negative_count'] + ['pc{}'.format(i + 1) for i in range(coords.shape[1])] + ['cluster']
    write_csv(output_dir / '{}_pca_coordinates.csv'.format(name), coord_rows, coord_fields)

    cluster_rows = summarize_clusters(rows, coords, labels)
    if cluster_rows:
        write_csv(output_dir / '{}_cluster_summary.csv'.format(name), cluster_rows,
                  ['cluster', 'n_users', 'mean_pc1', 'mean_pc2', 'mean_positive_count', 'mean_weak_negative_count', 'example_user_ids'])

    pairs = nearest_pairs(rows, matrix, nearest_top_n)
    write_csv(output_dir / '{}_nearest_user_pairs.csv'.format(name), pairs,
              ['user_id_a', 'user_id_b', 'cosine_similarity', 'a_positive_count', 'b_positive_count', 'a_weak_negative_count', 'b_weak_negative_count'])

    plot_scatter(output_dir / '{}_pca_scatter.png'.format(name), coords, rows, labels,
                 '{} PCA'.format(name.replace('_', ' ')), pca.explained_variance_ratio_)

    return {
        'space': name,
        'n_users': len(rows),
        'embedding_dim': int(matrix.shape[1]),
        'explained_variance_ratio': [float(x) for x in pca.explained_variance_ratio_],
        'cumulative_explained_variance': [float(x) for x in np.cumsum(pca.explained_variance_ratio_)],
        'best_silhouette': float(best_score) if best_score is not None else None,
        'k_scores': k_scores,
        'outputs': {
            'coordinates': str(output_dir / '{}_pca_coordinates.csv'.format(name)),
            'cluster_summary': str(output_dir / '{}_cluster_summary.csv'.format(name)),
            'nearest_pairs': str(output_dir / '{}_nearest_user_pairs.csv'.format(name)),
            'scatter': str(output_dir / '{}_pca_scatter.png'.format(name)),
        },
    }


def build_parser():
    parser = argparse.ArgumentParser(description='Analyze user profile embeddings with PCA and simple clustering.')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--n-components', type=int, default=10)
    parser.add_argument('--max-clusters', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260609)
    parser.add_argument('--nearest-top-n', type=int, default=25)
    parser.add_argument('--min-positive-count', type=int, default=1)
    parser.add_argument('--min-negative-count', type=int, default=1)
    return parser


def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, positive, negative, diff = load_embeddings(args.input, args.min_positive_count, args.min_negative_count)

    reports = []
    for name, matrix in [
        ('positive', positive),
        ('weak_negative', negative),
        ('positive_minus_negative', diff),
    ]:
        reports.append(analyze_space(name, matrix, rows, args.output_dir, args.n_components,
                                     args.max_clusters, args.seed, args.nearest_top_n))

    summary = {
        'input': str(args.input),
        'output_dir': str(args.output_dir),
        'n_users_used': len(rows),
        'filters': {
            'min_positive_count': args.min_positive_count,
            'min_negative_count': args.min_negative_count,
        },
        'spaces': reports,
    }
    with (args.output_dir / 'pca_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
