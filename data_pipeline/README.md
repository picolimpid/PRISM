# Bluesky User-Feed-Post Pipeline

This directory contains the pipeline used to construct user-feed-post interaction samples, build user-level text embeddings, and run PCA-based diagnostics.

## 1. User Sampling

Script:

```bash
/home/hanqim2/data_pipeline/build_user_feed_post_dataset.py
```

Main run:

```bash
python3 /home/hanqim2/data_pipeline/build_user_feed_post_dataset.py \
  --n-users 2000 \
  --max-feeds-per-user 10 \
  --negatives-per-feed 50 \
  --run-name first2000_user_feed_post_dataset \
  --overwrite
```

Users were sampled from filenames in:

```text
/srv/local2/common_resources/opal2/News_WellBeing/BlueSkyDataCopy/all_user_timelines/
```

The script parses files named `user_<id>.jsonl`, sorts them by numeric user id, and takes the first 2,000 files. This is not the same as selecting users with `user_id <= 2000`, because user ids are not continuous.

The script then scans each selected user's timeline file for records where `feed` is not null. These records are treated as evidence of an **active feed membership**. This is not a strict subscription/bookmark table. It means the user's timeline contains at least one feed-tagged event for that feed.

Output:

```text
/home/hanqim2/data_pipeline/first2000_user_feed_post_dataset/
```

Sample construction summary:

```text
selected timeline users: 2,000
users with active feed records: 549
user-feed pairs: 1,107
total user-feed-post rows: 74,901
positive rows: 20,653
weak negative rows: 54,248
```

## 2. Positive and Weak-Negative Post Classes

Each row in `user_feed_post_interactions.csv` is one `(user_id, feed, post_id)` example.

### Positive Class

A post is labeled `positive` for a user if the user had an observed interaction with that feed-tagged post. By default, the positive interaction types are:

```text
liker, commenter, reposter, quoter
```

The script groups these event types into one positive label while also keeping flags such as `liked`, `replied`, `reposted`, and `quoted`.

### Weak-Negative Class

Weak negatives are sampled from posts that appeared in the same active feed, but for which no interaction by that user was observed.

For each selected `(user_id, feed)` pair, the script reads:

```text
feed_posts/{feed}.jsonl
```

A post becomes a weak-negative candidate if:

```text
1. It belongs to that feed.
2. It is not in the user's positive post set.
3. It was not authored by the same user.
4. Its timestamp is not earlier than the user's first observed event in that feed.
```

Then the script randomly samples at most `--negatives-per-feed 50` posts from those candidates.

Important interpretation: these are **weak negatives**, not strict negatives. We only know that the user did not interact with the post in the observed data. We do not know whether the user actually saw the post.

## 3. Text Embedding

Embedding script:

```bash
/home/hanqim2/data_pipeline/build_label_embeddings.py
```

Environment:

```text
/home/hanqim2/.embed
Python 3.9.25
BAAI/bge-m3 via sentence-transformers
CUDA enabled
```

Run script:

```bash
/home/hanqim2/data_pipeline/run_bge_m3_embeddings.sh
```

Embedding output:

```text
/home/hanqim2/data_pipeline/first2000_user_feed_post_dataset/embeddings_bge_m3/
```

Files:

```text
user_label_embeddings.csv
global_label_embeddings.json
embedding_config.json
embedding_run.log
```

The embedding model input is cleaned `post_text` only. There is no natural-language prompt template, and the feed name is not explicitly included.

Config:

```text
include_feed_name: false
include_comment_text: false
keep_hashtags: true
model_name: BAAI/bge-m3
```

The script computes one embedding for each cleaned post text. It then averages embeddings by user and label:

```text
positive_embedding = mean embedding of all positive posts for that user
weak_negative_embedding = mean embedding of sampled weak-negative posts for that user
```

Embedding counts:

```text
processed rows: 72,909
skipped rows: 1,992
positive rows embedded: 20,209
weak negative rows embedded: 52,700
users in embedding table: 549
```

## 4. PCA Analysis

PCA script:

```bash
/home/hanqim2/data_pipeline/analyze_embedding_pca.py
```

Run:

```bash
/home/hanqim2/.embed/bin/python /home/hanqim2/data_pipeline/analyze_embedding_pca.py
```

PCA output:

```text
/home/hanqim2/data_pipeline/first2000_user_feed_post_dataset/embeddings_bge_m3/pca_analysis/
```

PCA was run on user-level embeddings, not on raw post embeddings. The analysis uses three vector spaces:

```text
positive
weak_negative
positive_minus_negative
```

### Plot Header Meanings

`positive_pca_scatter.png`

This PCA is run on each user's `positive_embedding`. It visualizes similarity among users based on the average content of posts they interacted with.

`weak_negative_pca_scatter.png`

This PCA is run on each user's `weak_negative_embedding`. It visualizes similarity among users based on the average content of sampled posts from their active feeds that they did not interact with.

`positive_minus_negative_pca_scatter.png`

This PCA is run on:

```text
positive_embedding - weak_negative_embedding
```

This difference vector is intended to approximate a user's preference direction: content they engaged with relative to content available in their active feeds but not engaged with.

In each PCA plot:

```text
x-axis = PC1
y-axis = PC2
point = one user
color = KMeans cluster label
point size = positive_count
```

Because point size is tied to `positive_count`, the plots can visually expose activity confounding.

## 5. Confounder Diagnostics

Diagnostic script:

```bash
/home/hanqim2/data_pipeline/diagnose_pca_confounders.py
```

Run:

```bash
/home/hanqim2/.embed/bin/python /home/hanqim2/data_pipeline/diagnose_pca_confounders.py
```

Diagnostic output:

```text
/home/hanqim2/data_pipeline/first2000_user_feed_post_dataset/embeddings_bge_m3/pca_analysis/confounder_diagnostics/
```

Files:

```text
cluster_activity_stats.csv
top_active_users.csv
cluster_feed_distribution.csv
cluster_feed_summary.csv
cluster_feed_js_divergence.csv
confounder_diagnostics_summary.json
```

Key finding: the preference clusters are strongly confounded by activity. Cluster 0 has much higher positive counts than cluster 1, even after removing the top 5% most active users.

## Suggested Next Step

Control for activity before interpreting preference clusters. A good next analysis is to match or stratify users by `positive_count`, then rerun PCA or clustering on `positive_minus_negative`.
