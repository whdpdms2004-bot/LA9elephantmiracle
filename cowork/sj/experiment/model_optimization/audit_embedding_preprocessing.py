from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler, normalize


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "experiment" / "model_optimization" / "pitcher_cluster_matchup"
sys.path.insert(0, str(WORK / "src"))

from cluster_profiles import eligible_mask, feature_columns  # noqa: E402


SEEDS = [17, 43, 97, 2026, 4099]


def transform(raw: np.ndarray, method: str, seed: int) -> tuple[np.ndarray, dict]:
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    filled = imputer.fit_transform(raw)
    if method.startswith("robust"):
        value = RobustScaler(quantile_range=(10.0, 90.0)).fit_transform(filled)
        cap = float(method.replace("robust", "")) if method != "robust_none" else None
        before = value.copy()
        if cap is not None:
            value = np.clip(value, -cap, cap)
        clipped = 0.0 if cap is None else float(np.not_equal(before, value).mean())
    elif method == "standard5":
        before = StandardScaler().fit_transform(filled)
        value = np.clip(before, -5.0, 5.0)
        clipped = float(np.not_equal(before, value).mean())
    elif method == "quantile_normal":
        value = QuantileTransformer(
            n_quantiles=min(100, len(raw)), output_distribution="normal",
            random_state=seed,
        ).fit_transform(filled)
        value = np.clip(value, -5.0, 5.0)
        clipped = np.nan
    else:
        raise ValueError(method)
    keep = np.nanstd(value, axis=0) > 1e-10
    return value[:, keep], {
        "input_features": int(raw.shape[1]),
        "kept_features": int(keep.sum()),
        "clipped_cell_rate": clipped,
        "max_row_norm_before_pca": float(np.linalg.norm(value[:, keep], axis=1).max()),
        "median_row_norm_before_pca": float(np.median(np.linalg.norm(value[:, keep], axis=1))),
    }


def fit_variant(part: pd.DataFrame, columns: list[str], k: int, method: str) -> tuple[np.ndarray, dict]:
    raw = part[columns].apply(pd.to_numeric, errors="coerce").to_numpy("float64")
    value, metrics = transform(raw, method, SEEDS[0])
    pca_dim = min(8, value.shape[1], len(value) - 1)
    embedding = PCA(n_components=pca_dim, random_state=SEEDS[0]).fit_transform(value)
    labels = []
    for seed in SEEDS:
        model = GaussianMixture(
            n_components=k, covariance_type="diag", reg_covar=1e-4,
            n_init=3, max_iter=500, random_state=seed,
        )
        labels.append(model.fit_predict(embedding))
    counts = np.bincount(labels[0], minlength=k)
    metrics.update({
        "pitchers": len(part),
        "k": k,
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "silhouette": float(silhouette_score(embedding, labels[0])),
        "seed_ari_mean": float(np.mean([
            adjusted_rand_score(labels[0], label) for label in labels[1:]
        ])),
        "pca_explained": float(
            PCA(n_components=pca_dim, random_state=SEEDS[0]).fit(value)
            .explained_variance_ratio_.sum()
        ),
    })
    return labels[0], metrics


def main() -> None:
    methods = ["robust_none", "robust5", "robust3", "standard5", "quantile_normal"]
    results = []
    column_rows = []
    labels_by_key = {}
    for cutoff in [2023, 2024, 2025]:
        profile = pd.read_parquet(WORK / "profiles" / f"pitcher_profile_cutoff_{cutoff}.parquet")
        columns = feature_columns(profile, "combined", 0.30)
        eligible = profile.loc[eligible_mask(profile, "combined")].copy()
        for column in columns:
            value = pd.to_numeric(eligible[column], errors="coerce")
            column_rows.append({
                "cutoff": cutoff, "column": column,
                "missing_rate": float(value.isna().mean()),
                "unique": int(value.nunique(dropna=True)),
                "q01": float(value.quantile(0.01)),
                "median": float(value.median()),
                "q99": float(value.quantile(0.99)),
            })
        for hand, k in [(1, 2), (2, 4)]:
            part = eligible.loc[eligible["pitcher_hand"].eq(hand)].copy()
            for method in methods:
                label, metrics = fit_variant(part, columns, k, method)
                key = (cutoff, hand, method)
                labels_by_key[key] = label
                results.append({
                    "cutoff": cutoff, "hand": hand, "method": method, **metrics
                })

    result = pd.DataFrame(results)
    comparisons = []
    for cutoff in [2023, 2024, 2025]:
        for hand in [1, 2]:
            anchor = labels_by_key[(cutoff, hand, "robust5")]
            for method in methods:
                comparisons.append({
                    "cutoff": cutoff, "hand": hand, "method": method,
                    "ari_vs_current_robust5": float(adjusted_rand_score(
                        anchor, labels_by_key[(cutoff, hand, method)]
                    )),
                })
    comparison = pd.DataFrame(comparisons)
    columns = pd.DataFrame(column_rows)
    result.to_csv(ROOT / "experiment/model_optimization/preprocess_embedding_variants.csv", index=False)
    comparison.to_csv(ROOT / "experiment/model_optimization/preprocess_embedding_variant_ari.csv", index=False)
    columns.to_csv(ROOT / "experiment/model_optimization/preprocess_embedding_columns.csv", index=False)
    print("VARIANTS")
    print(result.to_string(index=False))
    print("\nARI VS CURRENT")
    print(comparison.to_string(index=False))
    print("\nHIGHEST MISSING INCLUDED FEATURES")
    print(columns.sort_values("missing_rate", ascending=False).head(25).to_string(index=False))


if __name__ == "__main__":
    main()

