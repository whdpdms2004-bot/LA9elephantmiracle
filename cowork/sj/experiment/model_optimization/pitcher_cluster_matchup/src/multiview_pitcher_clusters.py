from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "experiment" / "model_optimization" / "pitcher_cluster_matchup"
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))

from cluster_profiles import (  # noqa: E402
    SEED,
    eligible_mask,
    feature_columns,
    make_embedding,
)


CUTOFFS = [2022, 2023, 2024, 2025]
ALGORITHMS = ["kmeans", "gmm_diag"]
VIEW_SPECS = [
    (8, 4, 0.5),
    (8, 4, 1.0),
    (8, 4, 2.0),
    (8, 8, 1.0),
    (16, 4, 1.0),
    (16, 8, 1.0),
]
K_PAIRS = [(2, 4), (3, 6), (4, 8), (6, 12)]
SEEDS = [2026, 2027, 2028, 2029, 2030]


def make_config(algorithm, phys_dim, control_dim, control_weight, k_pair):
    raw = f"{algorithm}|{phys_dim}|{control_dim}|{control_weight}|{k_pair}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    weight = str(control_weight).replace(".", "p")
    return (
        f"mv_{algorithm[:3]}_p{phys_dim}c{control_dim}w{weight}_"
        f"l{k_pair[0]}r{k_pair[1]}_{digest}"
    )


def fit_labels(embedding, algorithm, k, seed):
    if algorithm == "kmeans":
        return KMeans(n_clusters=k, n_init=30, random_state=seed).fit_predict(embedding)
    return GaussianMixture(
        n_components=k, covariance_type="diag", reg_covar=1e-4,
        n_init=5, max_iter=500, random_state=seed,
    ).fit_predict(embedding)


def build_embedding(part, physical_columns, control_columns, spec):
    phys_dim, control_dim, control_weight = spec
    physical, physical_explained = make_embedding(
        part, physical_columns, phys_dim, SEED
    )
    control, control_explained = make_embedding(
        part, control_columns, control_dim, SEED
    )
    # Equalize each latent coordinate before weighting the control view.
    physical = StandardScaler().fit_transform(physical)
    control = StandardScaler().fit_transform(control) * control_weight
    return np.column_stack([physical, control]), physical_explained, control_explained


def run_one(profile, cutoff, algorithm, spec, k_pair):
    physical_columns = feature_columns(profile, "physical", 0.30)
    control_columns = feature_columns(profile, "control", 0.30)
    eligible = eligible_mask(profile, "combined")
    selected = profile.loc[eligible].copy()
    pieces = []
    metrics = []
    for hand, k in [(1, k_pair[0]), (2, k_pair[1])]:
        part = selected.loc[selected["pitcher_hand"].eq(hand)].copy()
        embedding, phys_exp, control_exp = build_embedding(
            part, physical_columns, control_columns, spec
        )
        label_runs = [fit_labels(embedding, algorithm, k, seed) for seed in SEEDS]
        labels = label_runs[0].astype("int16")
        # Stable semantic ordering: velocity and control, not estimator label ID.
        ordering = []
        for old in range(k):
            member = labels == old
            speed = pd.to_numeric(
                part.loc[member, "tm500_recent_rel_speed_mean"], errors="coerce"
            ).median()
            control = pd.to_numeric(
                part.loc[member, "ctl_control_success_recent_resid"], errors="coerce"
            ).median()
            ordering.append((float(speed), float(control), old))
        mapping = {old: new for new, (_, _, old) in enumerate(sorted(ordering))}
        labels = np.asarray([mapping[int(value)] for value in labels], dtype="int16")
        output = part[["pitcher_id", "cutoff", "pitcher_hand", "cohort"]].copy()
        output["cluster_index"] = labels
        output["cluster_code"] = [f"H{hand}_MVC{x:02d}" for x in labels]
        pieces.append(output)
        counts = np.bincount(labels, minlength=k)
        ari = [adjusted_rand_score(label_runs[0], run) for run in label_runs[1:]]
        metrics.append({
            "hand": hand,
            "pitchers": int(len(part)),
            "min_cluster_size": int(counts.min()),
            "silhouette": float(silhouette_score(embedding, labels)),
            "seed_ari_mean": float(np.mean(ari)),
            "seed_ari_min": float(np.min(ari)),
            "physical_explained": phys_exp,
            "control_explained": control_exp,
        })
    assigned = pd.concat(pieces, ignore_index=True)
    fallback = profile.loc[
        ~profile["pitcher_id"].isin(assigned["pitcher_id"]),
        ["pitcher_id", "cutoff", "pitcher_hand", "cohort"],
    ].copy()
    fallback["cluster_index"] = -1
    # This explicitly keeps rookies/control-only pitchers out of TrackMan clusters.
    fallback["cluster_code"] = (
        "H" + fallback["pitcher_hand"].astype(str) + "_" + fallback["cohort"].astype(str)
    )
    lookup = pd.concat([assigned, fallback], ignore_index=True).sort_values("pitcher_id")
    config = make_config(algorithm, *spec, k_pair)
    weights = np.asarray([item["pitchers"] for item in metrics], dtype=float)
    weights /= weights.sum()
    row = {
        "config_id": config,
        "cutoff": cutoff,
        "representation": f"multiview_w{spec[2]}",
        "algorithm": algorithm,
        "pca_dim": spec[0] + spec[1],
        "physical_dim": spec[0],
        "control_dim": spec[1],
        "control_weight": spec[2],
        "k_left": k_pair[0],
        "k_right": k_pair[1],
        "eligible_pitchers": int(sum(item["pitchers"] for item in metrics)),
        "min_cluster_size": int(min(item["min_cluster_size"] for item in metrics)),
        "silhouette": float(np.dot(weights, [item["silhouette"] for item in metrics])),
        "seed_ari_mean": float(np.dot(weights, [item["seed_ari_mean"] for item in metrics])),
        "seed_ari_min": float(min(item["seed_ari_min"] for item in metrics)),
        "hand_metrics": json.dumps(metrics, ensure_ascii=False),
    }
    return config, lookup, row


def main():
    registry = []
    total = len(CUTOFFS) * len(ALGORITHMS) * len(VIEW_SPECS) * len(K_PAIRS)
    completed = 0
    for cutoff in CUTOFFS:
        profile = pd.read_parquet(
            WORK / "profiles" / f"pitcher_profile_cutoff_{cutoff}.parquet"
        )
        for algorithm in ALGORITHMS:
            for spec in VIEW_SPECS:
                for k_pair in K_PAIRS:
                    config, lookup, row = run_one(
                        profile, cutoff, algorithm, spec, k_pair
                    )
                    folder = WORK / "clusters" / config
                    folder.mkdir(parents=True, exist_ok=True)
                    lookup.to_parquet(folder / f"pitcher_lookup_{cutoff}.parquet", index=False)
                    registry.append(row)
                    completed += 1
                    print(json.dumps({
                        "completed": completed, "total": total, "config": config,
                        "cutoff": cutoff, "silhouette": row["silhouette"],
                        "seed_ari_mean": row["seed_ari_mean"],
                    }, ensure_ascii=False), flush=True)
    output = WORK / "reports" / "multiview_cluster_registry.csv"
    pd.DataFrame(registry).to_csv(output, index=False)
    print(json.dumps({"registry": str(output), "rows": len(registry)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
