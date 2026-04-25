from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install required packages with "
        "`pip install pandas scikit-learn matplotlib seaborn`."
    ) from exc

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pandas. Install required packages with "
        "`pip install pandas scikit-learn matplotlib seaborn`."
    ) from exc

try:
    import seaborn as sns
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: seaborn. Install required packages with "
        "`pip install pandas scikit-learn matplotlib seaborn`."
    ) from exc

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: scikit-learn. Install required packages with "
        "`pip install pandas scikit-learn matplotlib seaborn`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster a user feature matrix and save a 2D cluster visualization."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("2000user_feature_matrix.csv"),
        help="Path to the user-level feature matrix CSV.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of KMeans clusters to fit.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible clustering.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=6,
        help="Number of distinguishing attributes to print for each cluster.",
    )
    parser.add_argument(
        "--output-clustered-csv",
        type=Path,
        default=Path("clustered_user_features.csv"),
        help="Where to write the per-user cluster assignments.",
    )
    parser.add_argument(
        "--output-centroids-csv",
        type=Path,
        default=Path("cluster_centroids.csv"),
        help="Where to write the cluster centroid table on the original feature scale.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("cluster_plot.png"),
        help="Where to save the PCA cluster plot.",
    )
    return parser.parse_args()


def load_features(input_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_csv)
    if "user_id" not in df.columns:
        raise ValueError("The input CSV must contain a `user_id` column.")

    feature_df = df.drop(columns=[col for col in ["user_id", "cluster"] if col in df.columns]).copy()
    feature_df = feature_df.select_dtypes(include=["number"]).copy()
    if feature_df.empty:
        raise ValueError("No numeric feature columns were found after dropping `user_id`.")

    feature_df = feature_df.fillna(0.0)
    return df, feature_df


def fit_clusters(
    feature_df: pd.DataFrame, n_clusters: int, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, KMeans]:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(scaled_features)

    clustered_df = feature_df.copy()
    clustered_df["cluster"] = labels

    centroids = clustered_df.groupby("cluster", sort=True).mean(numeric_only=True)
    return clustered_df, centroids, scaler, kmeans


def print_cluster_summary(
    original_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
    centroids: pd.DataFrame,
    top_features: int,
) -> None:
    user_clusters = original_df[["user_id"]].copy()
    user_clusters["cluster"] = clustered_df["cluster"].values

    cluster_sizes = user_clusters["cluster"].value_counts().sort_index()
    print("\nCluster sizes")
    print(cluster_sizes.to_string())

    centroid_table = centroids.copy()
    centroid_table.insert(0, "user_count", cluster_sizes.reindex(centroid_table.index).values)
    pd.set_option("display.max_columns", None)
    print("\nCentroids on original feature scale")
    print(centroid_table.T)

    # Compare each centroid to the global feature distribution so we can print
    # the attributes that most strongly define each cluster.
    global_mean = feature_df.mean()
    global_std = feature_df.std(ddof=0).replace(0, 1.0)
    centroid_z = (centroids - global_mean) / global_std

    print("\nTop distinguishing attributes per cluster")
    for cluster_id in centroid_z.index:
        strongest = centroid_z.loc[cluster_id].abs().sort_values(ascending=False).head(top_features).index
        print(f"\nCluster {cluster_id}")
        for column in strongest:
            centroid_value = centroids.loc[cluster_id, column]
            z_value = centroid_z.loc[cluster_id, column]
            direction = "above" if z_value >= 0 else "below"
            print(
                f"  {column}: {centroid_value:.4f} "
                f"({abs(z_value):.2f} SD {direction} global mean)"
            )


def save_outputs(
    original_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
    centroids: pd.DataFrame,
    output_clustered_csv: Path,
    output_centroids_csv: Path,
) -> None:
    output_clustered_csv.parent.mkdir(parents=True, exist_ok=True)
    output_centroids_csv.parent.mkdir(parents=True, exist_ok=True)

    enriched_df = original_df.copy()
    enriched_df["cluster"] = clustered_df["cluster"].values
    enriched_df.to_csv(output_clustered_csv, index=False)
    centroids.to_csv(output_centroids_csv)


def save_cluster_plot(
    feature_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
    kmeans: KMeans,
    output_plot: Path,
) -> None:
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)

    pca = PCA(n_components=2, random_state=42)
    points_2d = pca.fit_transform(scaled_features)
    centroid_2d = pca.transform(kmeans.cluster_centers_)

    plot_df = pd.DataFrame(
        {
            "pc1": points_2d[:, 0],
            "pc2": points_2d[:, 1],
            "cluster": clustered_df["cluster"].astype(str),
        }
    )

    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=plot_df,
        x="pc1",
        y="pc2",
        hue="cluster",
        palette="tab10",
        alpha=0.75,
        s=36,
        linewidth=0,
    )
    plt.scatter(
        centroid_2d[:, 0],
        centroid_2d[:, 1],
        c="black",
        s=180,
        marker="X",
        label="Centroid",
    )
    plt.title("User Clusters (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    original_df, feature_df = load_features(args.input_csv)
    clustered_df, centroids, _, kmeans = fit_clusters(
        feature_df, args.n_clusters, args.random_state
    )

    print_cluster_summary(
        original_df=original_df,
        feature_df=feature_df,
        clustered_df=clustered_df,
        centroids=centroids,
        top_features=args.top_features,
    )
    save_outputs(
        original_df=original_df,
        clustered_df=clustered_df,
        centroids=centroids,
        output_clustered_csv=args.output_clustered_csv,
        output_centroids_csv=args.output_centroids_csv,
    )
    save_cluster_plot(
        feature_df=feature_df,
        clustered_df=clustered_df,
        kmeans=kmeans,
        output_plot=args.output_plot,
    )

    print(f"\nSaved clustered users to: {args.output_clustered_csv.resolve()}")
    print(f"Saved centroids to: {args.output_centroids_csv.resolve()}")
    print(f"Saved cluster plot to: {args.output_plot.resolve()}")


if __name__ == "__main__":
    main()
