#!/usr/bin/env python3
"""Diagnose activity and feed/topic confounding in PCA preference clusters."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_COORDS = Path('/home/hanqim2/data_pipeline/first2000_user_feed_post_dataset/embeddings_bge_m3/pca_analysis/positive_minus_negative_pca_coordinates.csv')
DEFAULT_FEEDS = Path('/home/hanqim2/data_pipeline/first2000_user_feed_post_dataset/user_feed_summary.csv')
DEFAULT_OUT = Path('/home/hanqim2/data_pipeline/first2000_user_feed_post_dataset/embeddings_bge_m3/pca_analysis/confounder_diagnostics')


def read_cluster_rows(path):
    rows = []
    with path.open('r', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            row['positive_count'] = int(row['positive_count'])
            row['weak_negative_count'] = int(row['weak_negative_count'])
            row['cluster'] = int(row['cluster'])
            rows.append(row)
    return rows


def read_feed_rows(path, user_cluster):
    rows = []
    with path.open('r', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            uid = row['user_id']
            if uid not in user_cluster:
                continue
            row['cluster'] = user_cluster[uid]
            row['positive_rows'] = int(row['positive_rows'])
            row['sampled_weak_negatives'] = int(row['sampled_weak_negatives'])
            rows.append(row)
    return rows


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def summarize_activity(rows):
    out = []
    for cluster in sorted(set(r['cluster'] for r in rows)):
        sub = [r for r in rows if r['cluster'] == cluster]
        pos = [r['positive_count'] for r in sub]
        neg = [r['weak_negative_count'] for r in sub]
        pos_sorted = sorted(pos)
        trim_n = max(1, int(math.ceil(len(pos_sorted) * 0.05))) if len(pos_sorted) > 20 else 0
        trimmed = pos_sorted[:-trim_n] if trim_n else pos_sorted
        out.append({
            'cluster': cluster,
            'n_users': len(sub),
            'positive_mean': sum(pos) / float(len(pos)),
            'positive_median': quantile(pos, 0.5),
            'positive_p25': quantile(pos, 0.25),
            'positive_p75': quantile(pos, 0.75),
            'positive_p90': quantile(pos, 0.9),
            'positive_p95': quantile(pos, 0.95),
            'positive_max': max(pos),
            'positive_mean_without_top5pct': sum(trimmed) / float(len(trimmed)) if trimmed else None,
            'weak_negative_mean': sum(neg) / float(len(neg)),
            'weak_negative_median': quantile(neg, 0.5),
            'weak_negative_max': max(neg),
        })
    return out


def top_active_users(rows, n):
    return sorted(rows, key=lambda r: (-r['positive_count'], r['user_id']))[:n]


def feed_distributions(feed_rows):
    cluster_user_counts = defaultdict(set)
    cluster_feed_users = defaultdict(Counter)
    cluster_feed_pos = defaultdict(Counter)
    cluster_feed_neg = defaultdict(Counter)

    for row in feed_rows:
        cluster = row['cluster']
        uid = row['user_id']
        feed = row['feed']
        cluster_user_counts[cluster].add(uid)
        cluster_feed_users[cluster][feed] += 1
        cluster_feed_pos[cluster][feed] += row['positive_rows']
        cluster_feed_neg[cluster][feed] += row['sampled_weak_negatives']

    rows = []
    for cluster in sorted(cluster_feed_users):
        n_users = len(cluster_user_counts[cluster])
        for feed, n_user_feed_pairs in cluster_feed_users[cluster].most_common():
            rows.append({
                'cluster': cluster,
                'feed': feed,
                'user_feed_pairs': n_user_feed_pairs,
                'share_of_cluster_users': n_user_feed_pairs / float(n_users) if n_users else 0.0,
                'positive_rows': cluster_feed_pos[cluster][feed],
                'sampled_weak_negatives': cluster_feed_neg[cluster][feed],
            })
    return rows


def entropy(counter):
    total = float(sum(counter.values()))
    if total <= 0:
        return 0.0
    h = 0.0
    for value in counter.values():
        p = value / total
        if p > 0:
            h -= p * math.log(p)
    return h


def js_divergence(counter_a, counter_b):
    feeds = set(counter_a) | set(counter_b)
    total_a = float(sum(counter_a.values()))
    total_b = float(sum(counter_b.values()))
    pa = {f: counter_a.get(f, 0) / total_a for f in feeds} if total_a else {f: 0.0 for f in feeds}
    pb = {f: counter_b.get(f, 0) / total_b for f in feeds} if total_b else {f: 0.0 for f in feeds}
    m = {f: 0.5 * (pa[f] + pb[f]) for f in feeds}

    def kl(p, q):
        val = 0.0
        for f in feeds:
            if p[f] > 0 and q[f] > 0:
                val += p[f] * math.log(p[f] / q[f])
        return val
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def feed_difference_summary(feed_rows):
    counters = defaultdict(Counter)
    for row in feed_rows:
        counters[row['cluster']][row['feed']] += 1
    clusters = sorted(counters)
    summary = []
    for cluster in clusters:
        total = sum(counters[cluster].values())
        summary.append({
            'cluster': cluster,
            'n_user_feed_pairs': total,
            'n_unique_feeds': len(counters[cluster]),
            'feed_entropy': entropy(counters[cluster]),
            'top_feeds': '|'.join('%s:%s' % (feed, count) for feed, count in counters[cluster].most_common(5)),
        })
    comparisons = []
    for i, a in enumerate(clusters):
        for b in clusters[i+1:]:
            comparisons.append({
                'cluster_a': a,
                'cluster_b': b,
                'js_divergence_feed_distribution': js_divergence(counters[a], counters[b]),
            })
    return summary, comparisons


def write_csv(path, rows, fieldnames):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coords', type=Path, default=DEFAULT_COORDS)
    parser.add_argument('--user-feed-summary', type=Path, default=DEFAULT_FEEDS)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--top-active-n', type=int, default=25)
    return parser


def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_rows = read_cluster_rows(args.coords)
    user_cluster = {row['user_id']: row['cluster'] for row in cluster_rows}
    feed_rows = read_feed_rows(args.user_feed_summary, user_cluster)

    activity = summarize_activity(cluster_rows)
    write_csv(args.output_dir / 'cluster_activity_stats.csv', activity,
              ['cluster', 'n_users', 'positive_mean', 'positive_median', 'positive_p25', 'positive_p75', 'positive_p90', 'positive_p95', 'positive_max', 'positive_mean_without_top5pct', 'weak_negative_mean', 'weak_negative_median', 'weak_negative_max'])

    active_fields = ['user_id', 'positive_count', 'weak_negative_count', 'pc1', 'pc2', 'cluster']
    active = [{k: row.get(k) for k in active_fields} for row in top_active_users(cluster_rows, args.top_active_n)]
    write_csv(args.output_dir / 'top_active_users.csv', active, active_fields)

    feeds = feed_distributions(feed_rows)
    write_csv(args.output_dir / 'cluster_feed_distribution.csv', feeds,
              ['cluster', 'feed', 'user_feed_pairs', 'share_of_cluster_users', 'positive_rows', 'sampled_weak_negatives'])

    feed_summary, feed_comparisons = feed_difference_summary(feed_rows)
    write_csv(args.output_dir / 'cluster_feed_summary.csv', feed_summary,
              ['cluster', 'n_user_feed_pairs', 'n_unique_feeds', 'feed_entropy', 'top_feeds'])
    write_csv(args.output_dir / 'cluster_feed_js_divergence.csv', feed_comparisons,
              ['cluster_a', 'cluster_b', 'js_divergence_feed_distribution'])

    payload = {
        'output_dir': str(args.output_dir),
        'n_users': len(cluster_rows),
        'n_user_feed_rows_joined': len(feed_rows),
        'activity_stats': activity,
        'feed_summary': feed_summary,
        'feed_js_divergence': feed_comparisons,
        'top_active_users_file': str(args.output_dir / 'top_active_users.csv'),
        'feed_distribution_file': str(args.output_dir / 'cluster_feed_distribution.csv'),
    }
    with (args.output_dir / 'confounder_diagnostics_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
