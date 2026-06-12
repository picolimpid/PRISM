#!/usr/bin/env python3
"""
Build a sampled user-feed-post interaction dataset from local Bluesky data.

Default behavior:
- Select first 2,000 users by numeric id from all_user_timelines/user_*.jsonl.
- Use liker/commenter/reposter/quoter events as positives.
- Infer user feeds from all_user_timelines by default.
- Sample weak negatives from feed_posts/{feed}.jsonl, capped per user-feed.

This script is Python 3.6 compatible and uses only the standard library.
"""

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT_DEFAULT = Path('/srv/local2/common_resources/opal2/News_WellBeing/BlueSkyDataCopy')
OUTPUT_DIR_DEFAULT = Path('/home/hanqim2/data_pipeline')
TIMELINE_USER_RE = re.compile(r'^user_(\d+)\.jsonl$')


class PositiveEvent(object):
    def __init__(self, user_id, feed, post_id):
        self.user_id = user_id
        self.feed = feed
        self.post_id = post_id
        self.first_interaction_ts = None
        self.post_content = None
        self.comment_content = None
        self.types = set()
        self.count = 0

    def add(self, exposure_type, ts, post_content, comment_content):
        self.types.add(exposure_type)
        self.count += 1
        if ts and (self.first_interaction_ts is None or comparable_ts(ts) < comparable_ts(self.first_interaction_ts)):
            self.first_interaction_ts = ts
        if self.post_content is None and post_content:
            self.post_content = post_content
        if self.comment_content is None and comment_content:
            self.comment_content = comment_content


class FeedPost(object):
    def __init__(self, feed, post_id, post_author_id, post_ts, post_ts_key, post_text,
                 like_count=None, reply_count=None, repost_count=None):
        self.feed = feed
        self.post_id = post_id
        self.post_author_id = post_author_id
        self.post_ts = post_ts
        self.post_ts_key = post_ts_key
        self.post_text = post_text
        self.like_count = like_count
        self.reply_count = reply_count
        self.repost_count = repost_count


def comparable_ts(value):
    parsed = parse_ts_key(value)
    return parsed if parsed is not None else 10 ** 18


def parse_ts_key(value):
    if value is None:
        return None
    if isinstance(value, float):
        value = int(value)
    text = str(value).strip()
    if not text:
        return None

    digits = re.sub(r'\D', '', text)
    if len(digits) >= 12:
        return int(digits[:12])
    if len(digits) == 8:
        return int(digits + '0000')
    return None


def format_ts_key(ts_key):
    if ts_key is None:
        return None
    text = '%012d' % ts_key
    try:
        return datetime.strptime(text, '%Y%m%d%H%M').strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        return text


def parse_date_arg(value):
    if value is None:
        return None
    parsed = parse_ts_key(value)
    if parsed is None:
        raise argparse.ArgumentTypeError('Could not parse date: %s' % value)
    return parsed


def safe_int(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def read_jsonl(path):
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get('type') == 'meta':
                continue
            yield record


def iter_timeline_files(timeline_dir):
    files = []
    for path in timeline_dir.iterdir():
        match = TIMELINE_USER_RE.match(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    return files


def choose_selected_users(timeline_dir, n_users):
    files = iter_timeline_files(timeline_dir)
    if len(files) < n_users:
        raise RuntimeError('Only found {:,} timeline files, fewer than requested {:,}.'.format(len(files), n_users))
    return files[:n_users]


def load_timeline_signals(selected_users, positive_types):
    positives = {}
    user_feed_counts = defaultdict(Counter)
    user_feed_start = {}

    for file_user_id, path in selected_users:
        for record in read_jsonl(path):
            feed = record.get('feed')
            if not feed:
                continue
            feed = str(feed)
            user_id = int(record.get('user_id', file_user_id))
            ts = record.get('timestamp')
            ts_key = parse_ts_key(ts)

            user_feed_counts[user_id][feed] += 1
            if ts_key is not None:
                key = (user_id, feed)
                if key not in user_feed_start or ts_key < user_feed_start[key]:
                    user_feed_start[key] = ts_key

            exposure_type = str(record.get('exposure_type') or '').lower()
            post_id = record.get('post_id')
            if exposure_type not in positive_types or post_id is None:
                continue
            try:
                post_id_int = int(post_id)
            except (TypeError, ValueError):
                continue

            key = (user_id, feed, post_id_int)
            if key not in positives:
                positives[key] = PositiveEvent(user_id, feed, post_id_int)
            positives[key].add(
                exposure_type,
                str(ts) if ts is not None else None,
                record.get('post_content'),
                record.get('comment_content')
            )

    return positives, user_feed_counts, user_feed_start


def load_bookmark_feeds(bookmarks_path, selected_user_ids):
    user_feed_counts = defaultdict(Counter)
    user_feed_start = {}
    with bookmarks_path.open('r', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            feed, user_id_raw, ts_raw = row[:3]
            try:
                user_id = int(user_id_raw)
            except ValueError:
                continue
            if user_id not in selected_user_ids:
                continue
            ts_key = parse_ts_key(ts_raw)
            user_feed_counts[user_id][feed] += 1
            if ts_key is not None:
                key = (user_id, feed)
                if key not in user_feed_start or ts_key < user_feed_start[key]:
                    user_feed_start[key] = ts_key
    return user_feed_counts, user_feed_start


def merge_feed_sources(base_counts, base_start, extra_counts, extra_start):
    for user_id, counts in extra_counts.items():
        base_counts[user_id].update(counts)
    for key, ts_key in extra_start.items():
        if key not in base_start or ts_key < base_start[key]:
            base_start[key] = ts_key


def select_user_feeds(selected_user_ids, user_feed_counts, max_feeds_per_user, feed_selection, rng):
    selected = {}
    for user_id in selected_user_ids:
        feeds = list(user_feed_counts.get(user_id, {}).items())
        if feed_selection == 'random':
            feed_names = [feed for feed, _ in feeds]
            rng.shuffle(feed_names)
        elif feed_selection == 'alphabetical':
            feed_names = sorted(feed for feed, _ in feeds)
        else:
            feed_names = [feed for feed, _ in sorted(feeds, key=lambda item: (-item[1], item[0]))]
        selected[user_id] = feed_names[:max_feeds_per_user]
    return selected


def load_feed_posts(feed_posts_dir, feeds, start_ts, end_ts):
    posts_by_feed = dict((feed, {}) for feed in feeds)
    for feed in sorted(feeds):
        path = feed_posts_dir / ('%s.jsonl' % feed)
        if not path.exists():
            continue
        for record in read_jsonl(path):
            post_id = record.get('post_id')
            if post_id is None:
                continue
            try:
                post_id_int = int(post_id)
            except (TypeError, ValueError):
                continue

            ts_key = parse_ts_key(record.get('date'))
            if start_ts is not None and (ts_key is None or ts_key < start_ts):
                continue
            if end_ts is not None and (ts_key is None or ts_key >= end_ts):
                continue

            current = posts_by_feed[feed].get(post_id_int)
            if current is not None and current.post_ts_key is not None and ts_key is not None and current.post_ts_key <= ts_key:
                continue

            author_id_int = safe_int(record.get('user_id'))
            posts_by_feed[feed][post_id_int] = FeedPost(
                feed=feed,
                post_id=post_id_int,
                post_author_id=author_id_int,
                post_ts=format_ts_key(ts_key),
                post_ts_key=ts_key,
                post_text=record.get('text'),
                like_count=safe_int(record.get('like_count')),
                reply_count=safe_int(record.get('reply_count')),
                repost_count=safe_int(record.get('repost_count')),
            )

    result = {}
    for feed, posts in posts_by_feed.items():
        result[feed] = sorted(posts.values(), key=lambda post: (post.post_ts_key is None, post.post_ts_key or 0, post.post_id))
    return result


def flags_for_types(types):
    return {
        'liked': int('liker' in types),
        'replied': int('commenter' in types),
        'reposted': int('reposter' in types),
        'quoted': int('quoter' in types),
        'posted': int('poster' in types),
    }


def write_selected_users(path, selected_users):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['rank', 'user_id', 'timeline_path'])
        writer.writeheader()
        for rank, item in enumerate(selected_users, start=1):
            user_id, timeline_path = item
            writer.writerow({'rank': rank, 'user_id': user_id, 'timeline_path': str(timeline_path)})


def make_row(user_id, feed, post_id, label, user_feed_start_ts, post, positive, seed):
    types = positive.types if positive else set()
    flags = flags_for_types(types)
    post_text = post.post_text if post is not None else None
    if post_text is None and positive is not None:
        post_text = positive.post_content

    row = {
        'user_id': user_id,
        'feed': feed,
        'post_id': post_id,
        'label': label,
        'engaged': int(label == 'positive'),
        'is_weak_negative': int(label == 'weak_negative'),
        'interaction_types': '|'.join(sorted(types)),
        'positive_event_count': positive.count if positive else 0,
        'first_interaction_ts': positive.first_interaction_ts if positive else None,
        'user_feed_start_ts': format_ts_key(user_feed_start_ts),
        'post_ts': post.post_ts if post is not None else None,
        'post_author_id': post.post_author_id if post is not None else None,
        'post_text': post_text,
        'comment_content_sample': positive.comment_content if positive else None,
        'post_like_count': post.like_count if post is not None else None,
        'post_reply_count': post.reply_count if post is not None else None,
        'post_repost_count': post.repost_count if post is not None else None,
        'negative_sampling_seed': seed if label == 'weak_negative' else None,
    }
    row.update(flags)
    return row


def write_dataset(output_path, selected_user_feeds, user_feed_start, positives, posts_by_feed,
                  negatives_per_feed, exclude_user_authored_negatives, require_post_after_feed_start, rng, seed):
    fieldnames = [
        'user_id', 'feed', 'post_id', 'label', 'engaged', 'is_weak_negative',
        'interaction_types', 'liked', 'replied', 'reposted', 'quoted', 'posted',
        'positive_event_count', 'first_interaction_ts', 'user_feed_start_ts',
        'post_ts', 'post_author_id', 'post_text', 'comment_content_sample',
        'post_like_count', 'post_reply_count', 'post_repost_count', 'negative_sampling_seed'
    ]

    positive_posts_by_user_feed = defaultdict(set)
    for user_id, feed, post_id in positives:
        positive_posts_by_user_feed[(user_id, feed)].add(post_id)

    post_lookup = {}
    for feed, posts in posts_by_feed.items():
        for post in posts:
            post_lookup[(feed, post.post_id)] = post

    stats = Counter()
    per_user_feed_rows = []
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for user_id in sorted(selected_user_feeds):
            for feed in selected_user_feeds[user_id]:
                start_ts = user_feed_start.get((user_id, feed))
                positive_post_ids = positive_posts_by_user_feed.get((user_id, feed), set())

                positive_rows = 0
                for post_id in sorted(positive_post_ids):
                    positive = positives[(user_id, feed, post_id)]
                    post = post_lookup.get((feed, post_id))
                    writer.writerow(make_row(user_id, feed, post_id, 'positive', start_ts, post, positive, seed))
                    stats['positive_rows'] += 1
                    positive_rows += 1

                negative_candidates = []
                for post in posts_by_feed.get(feed, []):
                    if post.post_id in positive_post_ids:
                        continue
                    if exclude_user_authored_negatives and post.post_author_id == user_id:
                        continue
                    if require_post_after_feed_start and start_ts is not None:
                        if post.post_ts_key is None or post.post_ts_key < start_ts:
                            continue
                    negative_candidates.append(post)

                if len(negative_candidates) > negatives_per_feed:
                    sampled = rng.sample(negative_candidates, negatives_per_feed)
                else:
                    sampled = negative_candidates

                for post in sampled:
                    writer.writerow(make_row(user_id, feed, post.post_id, 'weak_negative', start_ts, post, None, seed))
                    stats['weak_negative_rows'] += 1

                per_user_feed_rows.append({
                    'user_id': user_id,
                    'feed': feed,
                    'positive_rows': positive_rows,
                    'negative_candidates': len(negative_candidates),
                    'sampled_weak_negatives': len(sampled),
                })
                stats['user_feed_pairs'] += 1

    stats['total_rows'] = stats['positive_rows'] + stats['weak_negative_rows']
    stats['users_with_selected_feeds'] = sum(1 for feeds in selected_user_feeds.values() if feeds)
    return {'stats': dict(stats), 'per_user_feed_rows': per_user_feed_rows}


def write_summary(path, stats, per_user_feed_rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        fieldnames = ['user_id', 'feed', 'positive_rows', 'negative_candidates', 'sampled_weak_negatives']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_user_feed_rows)
    with path.with_name('summary.json').open('w', encoding='utf-8') as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)


def build_parser():
    parser = argparse.ArgumentParser(description='Create a sampled user-feed-post positive/weak-negative dataset.')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument('--timeline-dir', type=Path, default=None)
    parser.add_argument('--feed-posts-dir', type=Path, default=None)
    parser.add_argument('--bookmarks-path', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument('--run-name', default='first2000_user_feed_post_dataset')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--n-users', type=int, default=2000)
    parser.add_argument('--max-feeds-per-user', type=int, default=10)
    parser.add_argument('--negatives-per-feed', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20260609)
    parser.add_argument('--feed-source', choices=('timeline', 'bookmarks', 'both'), default='timeline')
    parser.add_argument('--feed-selection', choices=('count', 'random', 'alphabetical'), default='count')
    parser.add_argument('--positive-types', default='liker,commenter,reposter,quoter')
    parser.add_argument('--include-poster-positive', action='store_true')
    parser.add_argument('--start-date', type=parse_date_arg, default=None)
    parser.add_argument('--end-date', type=parse_date_arg, default=None)
    parser.add_argument('--exclude-user-authored-negatives', dest='exclude_user_authored_negatives', action='store_true', default=True)
    parser.add_argument('--include-user-authored-negatives', dest='exclude_user_authored_negatives', action='store_false')
    parser.add_argument('--require-post-after-feed-start', dest='require_post_after_feed_start', action='store_true', default=True)
    parser.add_argument('--allow-post-before-feed-start', dest='require_post_after_feed_start', action='store_false')
    return parser


def main():
    args = build_parser().parse_args()
    project_root = args.project_root
    timeline_dir = args.timeline_dir or project_root / 'all_user_timelines'
    feed_posts_dir = args.feed_posts_dir or project_root / 'feed_posts'
    bookmarks_path = args.bookmarks_path or project_root / 'feed_bookmarks.csv'
    run_dir = args.output_dir / args.run_name

    if run_dir.exists():
        if not args.overwrite:
            raise RuntimeError('Output directory already exists: %s. Use --overwrite or --run-name.' % run_dir)
        shutil.rmtree(str(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)

    positive_types = set(item.strip().lower() for item in args.positive_types.split(',') if item.strip())
    if args.include_poster_positive:
        positive_types.add('poster')

    rng = random.Random(args.seed)
    selected_users = choose_selected_users(timeline_dir, args.n_users)
    selected_user_ids = set(user_id for user_id, _ in selected_users)
    write_selected_users(run_dir / 'selected_users.csv', selected_users)

    positives, timeline_feed_counts, timeline_feed_start = load_timeline_signals(selected_users, positive_types)

    user_feed_counts = defaultdict(Counter)
    user_feed_start = {}
    if args.feed_source in ('timeline', 'both'):
        merge_feed_sources(user_feed_counts, user_feed_start, timeline_feed_counts, timeline_feed_start)
    if args.feed_source in ('bookmarks', 'both'):
        bookmark_counts, bookmark_start = load_bookmark_feeds(bookmarks_path, selected_user_ids)
        merge_feed_sources(user_feed_counts, user_feed_start, bookmark_counts, bookmark_start)

    selected_user_feeds = select_user_feeds(
        selected_user_ids, user_feed_counts, args.max_feeds_per_user, args.feed_selection, rng
    )
    selected_feeds = set(feed for feeds in selected_user_feeds.values() for feed in feeds)
    posts_by_feed = load_feed_posts(feed_posts_dir, selected_feeds, args.start_date, args.end_date)

    result = write_dataset(
        output_path=run_dir / 'user_feed_post_interactions.csv',
        selected_user_feeds=selected_user_feeds,
        user_feed_start=user_feed_start,
        positives=positives,
        posts_by_feed=posts_by_feed,
        negatives_per_feed=args.negatives_per_feed,
        exclude_user_authored_negatives=args.exclude_user_authored_negatives,
        require_post_after_feed_start=args.require_post_after_feed_start,
        rng=rng,
        seed=args.seed,
    )

    write_summary(run_dir / 'user_feed_summary.csv', result['stats'], result['per_user_feed_rows'])
    config = vars(args).copy()
    config.update({
        'project_root': str(project_root),
        'timeline_dir': str(timeline_dir),
        'feed_posts_dir': str(feed_posts_dir),
        'bookmarks_path': str(bookmarks_path),
        'run_dir': str(run_dir),
        'positive_types_resolved': sorted(positive_types),
        'selected_feed_count': len(selected_feeds),
    })
    with (run_dir / 'run_config.json').open('w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2, sort_keys=True, default=str)

    payload = {'run_dir': str(run_dir)}
    payload.update(result['stats'])
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
