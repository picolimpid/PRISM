from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pandas. Install required packages with "
        "`pip install pandas spacy textstat textblob`."
    ) from exc

try:
    import spacy
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: spacy. Install required packages with "
        "`pip install pandas spacy textstat textblob`."
    ) from exc

try:
    import textstat
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: textstat. Install required packages with "
        "`pip install pandas spacy textstat textblob`."
    ) from exc

try:
    from textblob import TextBlob
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: textblob. Install required packages with "
        "`pip install pandas spacy textstat textblob`."
    ) from exc


CRITICAL_COLUMNS = ["user_id", "timestamp", "exposure_type"]
OUTPUT_COLUMNS = [
    "user_id",
    "timestamp",
    "feed",
    "exposure_type",
    "post_id",
    "post_content",
    "comment_content",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a user-level feature matrix from Bluesky interaction JSONL files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/2000users"),
        help="Directory containing JSONL interaction files. Subdirectories are searched recursively.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("user_feature_matrix.csv"),
        help="Where to write the extracted feature matrix.",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy model to use for sentence segmentation/tokenization.",
    )
    return parser.parse_args()


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def safe_textstat(metric: Callable[[str], float], text: str) -> float:
    cleaned = " ".join(text.split())
    if not cleaned:
        return 0.0
    try:
        return float(metric(cleaned))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def load_interaction_data(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No JSONL files were found under {input_dir.resolve()}"
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_json(path, lines=True)
        if "type" in frame.columns:
            frame = frame[frame["type"].ne("meta")]
        if frame.empty:
            continue

        missing = [col for col in CRITICAL_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")

        for column in OUTPUT_COLUMNS:
            if column not in frame.columns:
                frame[column] = None

        frames.append(frame[OUTPUT_COLUMNS].copy())

    if not frames:
        raise ValueError(
            f"No interaction rows were found in {input_dir.resolve()}; the files only contained metadata."
        )

    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["user_id", "timestamp"]).copy()
    data["user_id"] = data["user_id"].astype(int)
    data["post_content"] = data["post_content"].fillna("").astype(str)
    data["comment_content"] = data["comment_content"].fillna("").astype(str)
    data["exposure_type"] = data["exposure_type"].fillna("").astype(str).str.lower()
    return data


def derive_authored_text(data: pd.DataFrame) -> pd.DataFrame:
    authored_text = pd.Series("", index=data.index, dtype="object")

    # Posters author the root post, while commenters and quoters author the reply text.
    poster_mask = data["exposure_type"].eq("poster")
    reply_mask = data["exposure_type"].isin(["commenter", "quoter"])

    authored_text.loc[poster_mask] = data.loc[poster_mask, "post_content"]
    authored_text.loc[reply_mask] = data.loc[reply_mask, "comment_content"]

    data = data.copy()
    data["authored_text"] = authored_text.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return data


def load_nlp(model_name: str) -> spacy.language.Language:
    try:
        nlp = spacy.load(model_name, exclude=["ner", "tagger", "lemmatizer", "attribute_ruler"])
    except OSError:
        nlp = spacy.blank("en")

    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def add_spacy_text_metrics(data: pd.DataFrame, nlp: spacy.language.Language) -> pd.DataFrame:
    data = data.copy()
    data["word_count"] = 0
    data["sentence_count"] = 0

    authored_mask = data["authored_text"].ne("")
    texts = data.loc[authored_mask, "authored_text"].tolist()
    if not texts:
        return data

    word_counts: list[int] = []
    sentence_counts: list[int] = []

    # spaCy handles the tokenization/sentence splitting so downstream counts
    # are based on a consistent NLP pipeline instead of plain whitespace splits.
    for doc in nlp.pipe(texts, batch_size=128):
        word_count = sum(1 for token in doc if not token.is_space and not token.is_punct)
        sentence_count = sum(1 for sent in doc.sents if sent.text.strip())
        word_counts.append(word_count)
        sentence_counts.append(sentence_count)

    data.loc[authored_mask, "word_count"] = word_counts
    data.loc[authored_mask, "sentence_count"] = sentence_counts
    return data


def liwc_placeholder(_: str) -> dict[str, float]:
    # Placeholder for future LIWC-style lexicon matching.
    # Return a dictionary of numeric features once a licensed LIWC dictionary
    # or an equivalent psycholinguistic lexicon is available.
    return {}


def compute_behavioral_features(data: pd.DataFrame) -> pd.DataFrame:
    def summarize(group: pd.DataFrame) -> pd.Series:
        timestamps = group["timestamp"].sort_values()
        gaps = timestamps.diff().dropna().dt.total_seconds().div(3600)
        span_days = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 86400 if len(timestamps) > 1 else 1.0
        active_days = max(span_days, 1.0)

        original_posts = int(group["exposure_type"].eq("poster").sum())
        reposts = int(group["exposure_type"].eq("reposter").sum())
        likes = int(group["exposure_type"].eq("liker").sum())
        replies = int(group["exposure_type"].eq("commenter").sum())

        return pd.Series(
            {
                "total_interactions": float(len(group)),
                "active_span_days": active_days,
                "post_frequency_per_day": len(group) / active_days,
                "original_post_count": float(original_posts),
                "repost_count": float(reposts),
                "like_count": float(likes),
                "reply_count": float(replies),
                "original_post_to_repost_ratio": safe_ratio(original_posts, reposts),
                "like_to_reply_ratio": safe_ratio(likes, replies),
                "avg_time_gap_hours": float(gaps.mean()) if not gaps.empty else 0.0,
            }
        )

    features = data.groupby("user_id", sort=True, group_keys=False).apply(summarize)
    features.index.name = "user_id"
    return features.reset_index()


def compute_text_features(data: pd.DataFrame) -> pd.DataFrame:
    authored = data[data["authored_text"].ne("")].copy()
    if authored.empty:
        return pd.DataFrame(columns=["user_id"])

    def summarize(group: pd.DataFrame) -> pd.Series:
        texts = group["authored_text"].tolist()
        joined_text = " ".join(texts)

        # TextBlob gives lightweight polarity/subjectivity scores that are easy
        # to average at the user level without extra model downloads.
        sentiments = [TextBlob(text).sentiment for text in texts]
        polarity = sum(item.polarity for item in sentiments) / len(sentiments)
        subjectivity = sum(item.subjectivity for item in sentiments) / len(sentiments)

        features = {
            "authored_text_count": float(len(texts)),
            "avg_word_count_per_post": float(group["word_count"].mean()),
            "avg_sentence_length": safe_ratio(group["word_count"].sum(), group["sentence_count"].sum()),
            "flesch_kincaid_grade": safe_textstat(textstat.flesch_kincaid_grade, joined_text),
            "flesch_reading_ease": safe_textstat(textstat.flesch_reading_ease, joined_text),
            "avg_polarity": float(polarity),
            "avg_subjectivity": float(subjectivity),
        }
        features.update(liwc_placeholder(joined_text))
        return pd.Series(features)

    features = authored.groupby("user_id", sort=True, group_keys=False).apply(summarize)
    features.index.name = "user_id"
    return features.reset_index()


def build_feature_matrix(input_dir: Path, spacy_model: str) -> pd.DataFrame:
    interactions = load_interaction_data(input_dir)
    interactions = derive_authored_text(interactions)
    nlp = load_nlp(spacy_model)
    interactions = add_spacy_text_metrics(interactions, nlp)

    behavioral = compute_behavioral_features(interactions)
    text_features = compute_text_features(interactions)

    feature_matrix = behavioral.merge(text_features, on="user_id", how="left")
    numeric_columns = [column for column in feature_matrix.columns if column != "user_id"]
    feature_matrix[numeric_columns] = feature_matrix[numeric_columns].fillna(0.0)
    return feature_matrix.sort_values("user_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    feature_matrix = build_feature_matrix(args.input_dir, args.spacy_model)
    feature_matrix.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(feature_matrix)} user rows to {args.output_csv.resolve()}")
    print(feature_matrix.head().to_string(index=False))


if __name__ == "__main__":
    main()
