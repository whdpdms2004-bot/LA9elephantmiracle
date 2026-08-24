from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "experiment" / "model_optimization" / "pitcher_cluster_matchup"
SEED = 2026
STYLE_FEATURES = [
    "tm500_recent_rel_speed_mean",
    "tm500_recent_zone_speed_mean",
    "tm500_recent_spin_rate_mean",
    "tm500_recent_induced_vert_break_mean",
    "tm500_recent_horz_break_mean_arm",
    "tm500_recent_extension_mean",
    "tm500_recent_rel_height_mean",
    "tm500_recent_rel_side_mean_arm",
    "tm500_recent_pitch_group_fastball_rate",
    "tm500_recent_pitch_group_breaking_rate",
    "tm500_recent_pitch_group_offspeed_rate",
    "tmg500_mix_recent_entropy",
    "tmg500_mix_recent_hhi",
    "ctl_control_success_recent_resid",
    "ctl_reverse_recent_resid",
    "ctl_middle_recent_resid",
    "ctl_outside_only_recent_resid",
    "ctl_batter_hand_resid_gap",
    "ctl_split_full_count_resid",
    "ctl_split_two_strike_resid",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoffs", default="2023,2024,2025")
    parser.add_argument("--representations", default="physical,combined")
    parser.add_argument("--algorithms", default="kmeans,gmm_diag")
    parser.add_argument("--pca-dims", default="8,16")
    parser.add_argument(
        "--k-pairs", default="2-4,3-6,4-8,5-10,6-12,8-16,8-20"
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--min-nonmissing", type=float, default=0.30)
    return parser.parse_args()


def feature_columns(profile, representation, min_nonmissing):
    control = [
        column for column in profile.columns
        if column.startswith("ctl_") and column not in {"ctl_rookie"}
        and not column.endswith("_n")
        and column not in {
            "ctl_total_n", "ctl_last_n", "ctl_last_season",
            "ctl_season_gap", "ctl_history_seasons",
        }
    ]
    physical = [
        column for column in profile.columns
        if column.startswith(("tm500_", "tmg500_"))
        and not any(token in column for token in [
            "available", "eligible_seasons", "total_pitches", "season_gap",
            "last_n", "last_season_n",
        ])
    ]
    if representation == "physical":
        candidates = physical
    elif representation == "control":
        candidates = control
    elif representation == "combined":
        candidates = physical + control
    else:
        raise ValueError(representation)
    keep = []
    for column in candidates:
        values = pd.to_numeric(profile[column], errors="coerce")
        if values.notna().mean() < min_nonmissing:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        keep.append(column)
    return keep


def eligible_mask(profile, representation):
    if representation in {"physical", "combined"}:
        return (
            profile["tm_available"].eq(1)
            & profile["ctl_rookie"].eq(0)
            & profile["ctl_season_gap"].le(2)
            & profile["tm500_season_gap"].le(3)
        )
    return (
        profile["ctl_total_n"].gt(100)
        & profile["ctl_rookie"].eq(0)
        & profile["ctl_season_gap"].le(2)
    )


def make_embedding(frame, columns, pca_dim, seed):
    raw = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy("float64")
    imputer = SimpleImputer(strategy="median", keep_empty_features=False)
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    transformed = scaler.fit_transform(imputer.fit_transform(raw))
    transformed = np.clip(transformed, -5.0, 5.0)
    finite_std = np.nanstd(transformed, axis=0)
    transformed = transformed[:, finite_std > 1e-10]
    effective_dim = max(1, min(pca_dim, transformed.shape[1], len(frame) - 1))
    pca = PCA(n_components=effective_dim, random_state=seed)
    embedding = pca.fit_transform(transformed)
    return embedding.astype("float32"), float(pca.explained_variance_ratio_.sum())


def fit_algorithm(embedding, algorithm, k, seed):
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = model.fit_predict(embedding)
        distance = model.transform(embedding)
        squared = distance ** 2
        scale = float(np.median(np.min(squared, axis=1)))
        scale = max(scale, 1e-4)
        logits = -squared / scale
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        centers = model.cluster_centers_
    elif algorithm == "gmm_diag":
        model = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=3,
            max_iter=500,
            random_state=seed,
        )
        labels = model.fit_predict(embedding)
        probability = model.predict_proba(embedding)
        centers = model.means_
        distance = np.linalg.norm(
            embedding[:, None, :] - centers[None, :, :], axis=2
        )
    else:
        raise ValueError(algorithm)
    assigned_distance = distance[np.arange(len(labels)), labels]
    return labels.astype("int16"), probability.astype("float32"), centers, assigned_distance


def fit_hand(part, columns, pca_dim, algorithm, k, seeds):
    embedding, explained = make_embedding(part, columns, pca_dim, seeds[0])
    labels_runs = []
    primary = None
    for seed in seeds:
        fitted = fit_algorithm(embedding, algorithm, k, seed)
        labels_runs.append(fitted[0])
        if primary is None:
            primary = fitted
    labels, probability, centers, _ = primary
    # Keep cluster numbers approximately comparable across cutoffs: order by
    # velocity first and control residual second, rather than estimator labels.
    speed_column = "tm500_recent_rel_speed_mean"
    control_column = "ctl_control_success_recent_resid"
    sort_records = []
    for old_label in range(k):
        members = labels == old_label
        speed = pd.to_numeric(part.loc[members, speed_column], errors="coerce").median()
        control = pd.to_numeric(part.loc[members, control_column], errors="coerce").median()
        sort_records.append((
            float(speed) if np.isfinite(speed) else -np.inf,
            float(control) if np.isfinite(control) else -np.inf,
            old_label,
        ))
    order = [item[2] for item in sorted(sort_records)]
    old_to_new = {old: new for new, old in enumerate(order)}
    labels = np.asarray([old_to_new[int(value)] for value in labels], dtype="int16")
    probability = probability[:, order]
    centers = centers[order]
    assigned_distance = np.linalg.norm(embedding - centers[labels], axis=1)
    ari = [adjusted_rand_score(labels_runs[0], values) for values in labels_runs[1:]]
    counts = np.bincount(labels, minlength=k)
    if len(np.unique(labels)) > 1 and len(labels) > k:
        silhouette = float(silhouette_score(embedding, labels))
        db = float(davies_bouldin_score(embedding, labels))
    else:
        silhouette = np.nan
        db = np.nan
    entropy = -np.sum(probability * np.log(np.clip(probability, 1e-8, 1.0)), axis=1)
    sorted_probability = np.sort(probability, axis=1)[:, ::-1]
    output = part[["pitcher_id", "cutoff", "pitcher_hand", "cohort"]].copy()
    output["cluster_index"] = labels
    output["cluster_code"] = [f"H{int(part['pitcher_hand'].iloc[0])}_C{int(v):02d}" for v in labels]
    output["cluster_distance"] = assigned_distance.astype("float32")
    output["cluster_entropy"] = entropy.astype("float32")
    output["cluster_top_gap"] = (
        sorted_probability[:, 0] - sorted_probability[:, 1]
    ).astype("float32")
    output["cluster_size_pitchers"] = counts[labels].astype("int16")
    for rank in range(min(5, probability.shape[1])):
        output[f"cluster_q_rank{rank + 1}"] = sorted_probability[:, rank]
    for dimension in range(embedding.shape[1]):
        output[f"cluster_emb_{dimension:02d}"] = embedding[:, dimension]
        output[f"cluster_center_{dimension:02d}"] = centers[labels, dimension]
    for column in STYLE_FEATURES:
        if column not in part:
            continue
        numeric = pd.to_numeric(part[column], errors="coerce")
        cluster_median = numeric.groupby(labels).median()
        output[f"cluster_style_{column}"] = pd.Series(labels).map(cluster_median).to_numpy("float32")
    metrics = {
        "pitchers": int(len(part)),
        "k": int(k),
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "silhouette": silhouette,
        "davies_bouldin": db,
        "seed_ari_mean": float(np.mean(ari)) if ari else 1.0,
        "seed_ari_min": float(np.min(ari)) if ari else 1.0,
        "pca_explained": explained,
        "embedding_dim": int(embedding.shape[1]),
        "feature_count": int(len(columns)),
    }
    return output, metrics


def config_id(representation, algorithm, pca_dim, k_left, k_right):
    raw = f"{representation}|{algorithm}|p{pca_dim}|l{k_left}|r{k_right}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{representation[:3]}_{algorithm[:3]}_p{pca_dim}_l{k_left}r{k_right}_{digest}"


def run_config(profile, cutoff, representation, algorithm, pca_dim, k_pair, seeds, min_nonmissing):
    columns = feature_columns(profile, representation, min_nonmissing)
    eligible = eligible_mask(profile, representation)
    selected = profile.loc[eligible].copy()
    k_by_hand = {1: k_pair[0], 2: k_pair[1]}
    pieces = []
    hand_metrics = []
    for hand, k in k_by_hand.items():
        part = selected.loc[selected["pitcher_hand"].eq(hand)].copy()
        if len(part) <= k or len(part) < 2 * k:
            raise ValueError(f"cutoff={cutoff} hand={hand}: n={len(part)} too small for k={k}")
        lookup, metrics = fit_hand(part, columns, pca_dim, algorithm, k, seeds)
        metrics["hand"] = hand
        hand_metrics.append(metrics)
        pieces.append(lookup)

    assigned = pd.concat(pieces, ignore_index=True)
    fallback = profile.loc[~profile["pitcher_id"].isin(assigned["pitcher_id"]), [
        "pitcher_id", "cutoff", "pitcher_hand", "cohort",
    ]].copy()
    fallback["cluster_index"] = -1
    fallback["cluster_code"] = (
        "H" + fallback["pitcher_hand"].astype(str) + "_" + fallback["cohort"].astype(str)
    )
    for column in assigned.columns:
        if column not in fallback:
            fallback[column] = np.nan
    lookup = pd.concat([assigned, fallback[assigned.columns]], ignore_index=True)
    lookup = lookup.sort_values("pitcher_id").reset_index(drop=True)
    config = config_id(representation, algorithm, pca_dim, *k_pair)
    lookup.insert(0, "config_id", config)
    weights = np.array([item["pitchers"] for item in hand_metrics], dtype="float64")
    weights /= weights.sum()
    aggregate = {
        "config_id": config,
        "cutoff": cutoff,
        "representation": representation,
        "algorithm": algorithm,
        "pca_dim": pca_dim,
        "k_left": k_pair[0],
        "k_right": k_pair[1],
        "eligible_pitchers": int(sum(item["pitchers"] for item in hand_metrics)),
        "min_cluster_size": int(min(item["min_cluster_size"] for item in hand_metrics)),
        "silhouette": float(np.dot(weights, [item["silhouette"] for item in hand_metrics])),
        "davies_bouldin": float(np.dot(weights, [item["davies_bouldin"] for item in hand_metrics])),
        "seed_ari_mean": float(np.dot(weights, [item["seed_ari_mean"] for item in hand_metrics])),
        "seed_ari_min": float(min(item["seed_ari_min"] for item in hand_metrics)),
        "pca_explained": float(np.dot(weights, [item["pca_explained"] for item in hand_metrics])),
        "feature_count": int(len(columns)),
        "hand_metrics": json.dumps(hand_metrics, ensure_ascii=False),
    }
    return config, lookup, aggregate


def main():
    args = parse_args()
    cutoffs = [int(v) for v in args.cutoffs.split(",") if v]
    representations = [v for v in args.representations.split(",") if v]
    algorithms = [v for v in args.algorithms.split(",") if v]
    pca_dims = [int(v) for v in args.pca_dims.split(",") if v]
    k_pairs = [tuple(map(int, value.split("-"))) for value in args.k_pairs.split(",") if value]
    seeds = [SEED + index for index in range(args.seeds)]
    registry = []
    cluster_dir = WORK / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    for cutoff in cutoffs:
        profile = pd.read_parquet(WORK / "profiles" / f"pitcher_profile_cutoff_{cutoff}.parquet")
        for representation in representations:
            for algorithm in algorithms:
                for pca_dim in pca_dims:
                    for k_pair in k_pairs:
                        try:
                            config, lookup, row = run_config(
                                profile, cutoff, representation, algorithm, pca_dim,
                                k_pair, seeds, args.min_nonmissing,
                            )
                        except ValueError as error:
                            print(json.dumps({"skipped": str(error)}, ensure_ascii=False), flush=True)
                            continue
                        folder = cluster_dir / config
                        folder.mkdir(parents=True, exist_ok=True)
                        lookup.to_parquet(folder / f"pitcher_lookup_{cutoff}.parquet", index=False)
                        registry.append(row)
                        print(json.dumps({
                            key: row[key] for key in [
                                "config_id", "cutoff", "eligible_pitchers",
                                "min_cluster_size", "silhouette", "seed_ari_mean",
                            ]
                        }, ensure_ascii=False), flush=True)
    registry_frame = pd.DataFrame(registry)
    registry_path = WORK / "reports" / "cluster_registry.csv"
    registry_frame.to_csv(registry_path, index=False)
    summary = (
        registry_frame.groupby([
            "config_id", "representation", "algorithm", "pca_dim", "k_left", "k_right"
        ], as_index=False)
        .agg(
            cutoff_count=("cutoff", "nunique"),
            min_cluster_size=("min_cluster_size", "min"),
            silhouette=("silhouette", "mean"),
            seed_ari_mean=("seed_ari_mean", "mean"),
            seed_ari_min=("seed_ari_min", "min"),
            pca_explained=("pca_explained", "mean"),
        )
    )
    summary["screen_score"] = (
        summary["silhouette"]
        + 0.25 * summary["seed_ari_mean"]
        + 0.02 * np.minimum(summary["min_cluster_size"], 10)
    )
    summary = summary.sort_values(
        ["min_cluster_size", "screen_score"], ascending=[False, False]
    )
    summary_path = WORK / "reports" / "cluster_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(json.dumps({
        "registry": str(registry_path),
        "summary": str(summary_path),
        "configs": int(summary["config_id"].nunique()),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
