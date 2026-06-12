#!/usr/bin/env bash
set -euo pipefail

OUTDIR=/home/hanqim2/data_pipeline/first2000_user_feed_post_dataset/embeddings_bge_m3
mkdir -p "$OUTDIR"

/home/hanqim2/.embed/bin/python /home/hanqim2/data_pipeline/build_label_embeddings.py \
  --dataset /home/hanqim2/data_pipeline/first2000_user_feed_post_dataset/user_feed_post_interactions.csv \
  --output-dir "$OUTDIR" \
  --model-name BAAI/bge-m3 \
  --device cuda \
  --text-batch-size 1024 \
  --encode-batch-size 32 \
  --show-progress \
  2>&1 | tee "$OUTDIR/embedding_run.log"
