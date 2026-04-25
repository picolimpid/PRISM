# How to obtain data from server
```bash
mkdir -p /home/hanqim2/2000users
find /srv/local2/common_resources/opal2/News_WellBeing/BlueSkyDataCopy/all_user_timelines \
  -maxdepth 1 -type f -name 'user_*.jsonl' | sort -V | head -n 2000 | \
  xargs -I{} cp "{}" /home/hanqim2/2000users/
```
```bash
find /home/hanqim2/2000users -maxdepth 1 -type f | wc -l
```
```bash
scp -r oncare-prism:/home/hanqim2/2000users C:\Users\andym\Downloads\Prism\data
```
# Workflow

## Files in `src/`

### `src/extract_user_features.py`

Reads Bluesky interaction `.jsonl` files, aggregates them at the user level, and writes a feature matrix.

What it extracts:
- Behavioral signals such as post frequency, repost ratio, like/reply ratio, and interaction timing
- Text signals such as average word count, sentence length, readability, polarity, and subjectivity

Current default input:
- `data/2000users/`

Typical output:
- `2000user_feature_matrix.csv`

### `src/cluster_user_features.py`

Reads a feature matrix CSV, standardizes numeric features with `StandardScaler`, runs `KMeans`, prints a cluster summary, and saves clustering outputs.

Typical outputs:
- `cluster_result/5centroid/2000user_feature_matrix_clustered.csv`
- `cluster_result/5centroid/2000user_cluster_centroids.csv`
- `cluster_result/5centroid/2000user_cluster_plot.png`

### `src/app.py`

Runs the Streamlit radar-chart dashboard for the 5 baseline personas.

Note:
- The chart data is currently hard-coded to the 5 baseline personas
- It does not yet read directly from the generated CSV files

## Method

### Step 1: Multi-Dimensional Feature Engineering

How it was done:
Leveraging the methodological toolkit from the Emails by LLMs paper, this workflow avoids scraping invasive user data. Instead, it condenses the fragmented posting histories of 2,000 real Bluesky users into 17 high-density feature vectors.

Scope:
These features cover behavioral patterns such as post frequency and repost ratio, syntactic structure such as Flesch reading ease and average word count, and surface-level semantics such as average polarity and subjectivity.

### Step 2: Data-Driven Unsupervised Clustering

How it was done:
Following the approach from the Observer Effect paper, the clustering step explicitly avoids demographic labels. Numeric features are first normalized with `StandardScaler`, then clustered with `KMeans`.

I use 6 of 17 attributes of each cluster to demonstrate core difference(Borrow idea from Professor's two paper, Observer Effect and Email by LLM):

Dimensional Mapping & Interpretations
Activity: Maps to post_frequency_per_day. This reflects whether a user is a high-volume "broadcaster" or a "low-frequency" casual user.

Engagement: Maps to reply_count. This reflects whether a user prefers participating in two-way discussions or relies on one-way content output.

Originality: Maps to original_post_to_repost_ratio. This distinguishes genuine content creators from information aggregators/amplifiers.

Subjectivity: Maps to avg_subjectivity. This reflects whether a user's discourse is primarily fact-based (objective) or driven by personal opinions and emotions.

Readability: Maps to flesch_reading_ease. This reflects the linguistic style of the user, indicating whether their text is casual/conversational or complex/academic.

Polarity: Maps to avg_polarity. This reflects the baseline emotional tone of the user, indicating whether they skew positive and cheerful or neutral and cold.
Key results:

Using the elbow-method and silhouette-score workflow, the current baseline converges on `K=5`. This yields 5 distinct social media personas:
- The Everyday Creator
- The Heavy Debater
- The Silent Lurker
- The Pure Curator
- The Hyper-Spreader

At this stage, the behavioral skeleton of the agents is in place.

### Step 3: Semantic Extraction of Topics and Values (In Progress)

How it is being done:
Behavioral data explains how users act, but not what they talk about. The current step samples `authored_text` from the 5 clusters and feeds that text into an LLM.

Objective:
Infer 3 to 5 thematic categories and underlying social values for each group, such as logic-driven discussion versus emotional resonance. Once completed, the agents will have both behavioral structure and semantic identity.

## Setup

The commands below assume you are running them from the repository root in Git Bash.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Recommended spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Run

The commands below are written for Git Bash.

### 1. Build the feature matrix

Run on the 2,000-user dataset:

```bash
python src/extract_user_features.py --input-dir data/2000users --output-csv 2000user_feature_matrix.csv
```

Small test run on the 20-user dataset:

```bash
python src/extract_user_features.py --input-dir data/20users --output-csv user_feature_matrix.csv
```

### 2. Cluster the feature matrix

Baseline 5-cluster run:

```bash
python src/cluster_user_features.py \
  --input-csv 2000user_feature_matrix.csv \
  --n-clusters 5 \
  --output-clustered-csv cluster_result/5centroid/2000user_feature_matrix_clustered.csv \
  --output-centroids-csv cluster_result/5centroid/2000user_cluster_centroids.csv \
  --output-plot cluster_result/5centroid/2000user_cluster_plot.png
```

### 3. Launch the dashboard

```bash
streamlit run src/app.py
```

## Notes

- `data/`, `papers/`, and `.prism/` are ignored by Git
- `cluster_result/` is intentionally tracked, including generated CSV and PNG outputs
- Root-level matrix files such as `user_feature_matrix.csv` and `2000user_feature_matrix.csv` are intentionally kept
