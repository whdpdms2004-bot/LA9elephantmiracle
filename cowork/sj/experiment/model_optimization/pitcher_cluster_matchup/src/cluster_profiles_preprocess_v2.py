from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer, RobustScaler


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "experiment" / "model_optimization" / "pitcher_cluster_matchup"
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))

from cluster_profiles import STYLE_FEATURES, eligible_mask, feature_columns  # noqa: E402


PROFILE_DIR = WORK / "profiles_clean_v2"
CLUSTER_DIR = WORK / "clusters_preprocess_v2"
REPORTS = WORK / "reports"
CUTOFFS = [2022, 2023, 2024, 2025]
SEEDS = [2026, 2027, 2028, 2029, 2030]


def feature_contracts(profiles: dict[int, pd.DataFrame]) -> dict[str, list[str]]:
    current_sets = {
        cutoff: set(feature_columns(profile, "combined", 0.30))
        for cutoff, profile in profiles.items()
    }
    all_current = sorted(set.intersection(*current_sets.values()))
    eligible_stable = []
    for column in all_current:
        stable = True
        for profile in profiles.values():
            part = profile.loc[eligible_mask(profile, "combined")]
            value = pd.to_numeric(part[column], errors="coerce")
            if value.notna().mean() < 0.70 or value.nunique(dropna=True) <= 1:
                stable = False
                break
        if stable:
            eligible_stable.append(column)
    compact = [
        column for column in STYLE_FEATURES
        if all(column in profile.columns for profile in profiles.values())
    ]
    physical_compact = [
        column for column in compact if column.startswith(("tm500_", "tmg500_"))
    ]
    return {
        "all": all_current,
        "stable": eligible_stable,
        "compact": compact,
        "physical": physical_compact,
    }


def candidate_specs() -> list[dict]:
    specs = [
        {"name": "all_r5", "contract": "all", "method": "robust5", "missing": False, "k": (2, 4)},
        {"name": "stable_r5", "contract": "stable", "method": "robust5", "missing": False, "k": (2, 4)},
        {"name": "compact_r5", "contract": "compact", "method": "robust5", "missing": False, "k": (2, 4)},
        {"name": "compact_mi_r5", "contract": "compact", "method": "robust5", "missing": True, "k": (2, 4)},
        {"name": "compact_mi_r3", "contract": "compact", "method": "robust3", "missing": True, "k": (2, 4)},
        {"name": "compact_mi_qn", "contract": "compact", "method": "quantile", "missing": True, "k": (2, 4)},
        {"name": "physical_mi_r5", "contract": "physical", "method": "robust5", "missing": True, "k": (2, 4)},
    ]
    for k in [(3, 6), (4, 8)]:
        specs.extend([
            {"name": f"compact_mi_r5_k{k[0]}{k[1]}", "contract": "compact", "method": "robust5", "missing": True, "k": k},
            {"name": f"stable_r5_k{k[0]}{k[1]}", "contract": "stable", "method": "robust5", "missing": False, "k": k},
        ])
    return specs


def config_id(spec: dict) -> str:
    raw = json.dumps(spec, sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"pv2_{spec['name']}_{digest}"


def preprocess(raw: np.ndarray, method: str, add_missing: bool, seed: int):
    missing = ~np.isfinite(raw)
    value = SimpleImputer(strategy="median").fit_transform(raw)
    if method.startswith("robust"):
        value = RobustScaler(quantile_range=(10.0, 90.0)).fit_transform(value)
        cap = float(method.replace("robust", ""))
        value = np.clip(value, -cap, cap)
    elif method == "quantile":
        value = QuantileTransformer(
            n_quantiles=min(100, len(value)), output_distribution="normal",
            random_state=seed,
        ).fit_transform(value)
        value = np.clip(value, -5.0, 5.0)
    else:
        raise ValueError(method)
    if add_missing:
        useful = missing.any(axis=0) & ~missing.all(axis=0)
        if useful.any():
            value = np.column_stack([value, missing[:, useful].astype("float64")])
    keep = np.nanstd(value, axis=0) > 1e-10
    return value[:, keep]


def fit_hand(part: pd.DataFrame, columns: list[str], k: int, spec: dict):
    raw = part[columns].apply(pd.to_numeric, errors="coerce").to_numpy("float64")
    value = preprocess(raw, spec["method"], spec["missing"], SEEDS[0])
    dim = max(1, min(8, value.shape[1], len(value) - 1))
    pca = PCA(n_components=dim, random_state=SEEDS[0])
    embedding = pca.fit_transform(value)
    labels_runs = []
    primary_model = None
    for seed in SEEDS:
        model = GaussianMixture(
            n_components=k, covariance_type="diag", reg_covar=1e-4,
            n_init=5, max_iter=500, random_state=seed,
        )
        labels_runs.append(model.fit_predict(embedding))
        if primary_model is None:
            primary_model = model
    labels = labels_runs[0]
    speed = pd.to_numeric(part.get("tm500_recent_rel_speed_mean"), errors="coerce")
    control = pd.to_numeric(part.get("ctl_control_success_recent_resid"), errors="coerce")
    order = sorted(
        range(k),
        key=lambda label: (
            float(speed[labels == label].median()),
            float(control[labels == label].median()),
        ),
    )
    mapping = {old: new for new, old in enumerate(order)}
    labels = np.asarray([mapping[int(label)] for label in labels], dtype="int16")
    counts = np.bincount(labels, minlength=k)
    lookup = part[["pitcher_id", "cutoff", "pitcher_hand", "cohort"]].copy()
    lookup["cluster_index"] = labels
    lookup["cluster_code"] = [
        f"PV2H{int(part['pitcher_hand'].iloc[0])}_C{int(label):02d}" for label in labels
    ]
    metrics = {
        "pitchers": len(part), "k": k,
        "min_cluster_size": int(counts.min()), "max_cluster_size": int(counts.max()),
        "silhouette": float(silhouette_score(embedding, labels)),
        "davies_bouldin": float(davies_bouldin_score(embedding, labels)),
        "seed_ari_mean": float(np.mean([
            adjusted_rand_score(labels_runs[0], other) for other in labels_runs[1:]
        ])),
        "seed_ari_min": float(np.min([
            adjusted_rand_score(labels_runs[0], other) for other in labels_runs[1:]
        ])),
        "pca_explained": float(pca.explained_variance_ratio_.sum()),
        "input_features": len(columns), "transformed_features": value.shape[1],
    }
    return lookup, metrics


def main() -> None:
    profiles = {
        cutoff: pd.read_parquet(PROFILE_DIR / f"pitcher_profile_cutoff_{cutoff}.parquet")
        for cutoff in CUTOFFS
    }
    contracts = feature_contracts(profiles)
    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    contract_rows = []
    for name, columns in contracts.items():
        contract_rows.extend({"contract": name, "column": column} for column in columns)
    for spec in candidate_specs():
        config = config_id(spec)
        columns = contracts[spec["contract"]]
        for cutoff, profile in profiles.items():
            selected = profile.loc[eligible_mask(profile, "combined")].copy()
            hand_rows = []
            pieces = []
            for hand, k in [(1, spec["k"][0]), (2, spec["k"][1])]:
                part = selected.loc[selected["pitcher_hand"].eq(hand)].copy()
                lookup, metrics = fit_hand(part, columns, k, spec)
                metrics["hand"] = hand
                hand_rows.append(metrics)
                pieces.append(lookup)
            assigned = pd.concat(pieces, ignore_index=True)
            fallback = profile.loc[
                ~profile["pitcher_id"].isin(assigned["pitcher_id"]),
                ["pitcher_id", "cutoff", "pitcher_hand", "cohort"],
            ].copy()
            fallback["cluster_index"] = -1
            fallback["cluster_code"] = (
                "PV2H" + fallback["pitcher_hand"].astype(str)
                + "_" + fallback["cohort"].astype(str)
            )
            full = pd.concat([assigned, fallback], ignore_index=True).sort_values("pitcher_id")
            full.insert(0, "config_id", config)
            folder = CLUSTER_DIR / config
            folder.mkdir(parents=True, exist_ok=True)
            full.to_parquet(folder / f"pitcher_lookup_{cutoff}.parquet", index=False)
            weights = np.asarray([item["pitchers"] for item in hand_rows], dtype=float)
            weights /= weights.sum()
            row = {
                "config_id": config, "cutoff": cutoff,
                "representation": f"pv2_{spec['contract']}_{spec['method']}",
                "algorithm": "gmm_diag", "pca_dim": 8,
                "k_left": spec["k"][0], "k_right": spec["k"][1],
                "eligible_pitchers": int(sum(item["pitchers"] for item in hand_rows)),
                "min_cluster_size": int(min(item["min_cluster_size"] for item in hand_rows)),
                "silhouette": float(np.dot(weights, [item["silhouette"] for item in hand_rows])),
                "davies_bouldin": float(np.dot(weights, [item["davies_bouldin"] for item in hand_rows])),
                "seed_ari_mean": float(np.dot(weights, [item["seed_ari_mean"] for item in hand_rows])),
                "seed_ari_min": float(min(item["seed_ari_min"] for item in hand_rows)),
                "pca_explained": float(np.dot(weights, [item["pca_explained"] for item in hand_rows])),
                "feature_count": len(columns), "spec": json.dumps(spec, ensure_ascii=False),
                "hand_metrics": json.dumps(hand_rows, ensure_ascii=False),
            }
            rows.append(row)
            print(json.dumps({
                "config": config, "cutoff": cutoff,
                "features": len(columns), "min_cluster": row["min_cluster_size"],
                "seed_ari": row["seed_ari_mean"],
            }, ensure_ascii=False), flush=True)
    registry = pd.DataFrame(rows)
    registry.to_csv(REPORTS / "cluster_registry_preprocess_v2.csv", index=False)
    pd.DataFrame(contract_rows).to_csv(
        REPORTS / "cluster_feature_contracts_preprocess_v2.csv", index=False
    )
    print(json.dumps({
        "registry": str(REPORTS / "cluster_registry_preprocess_v2.csv"),
        "configs": registry["config_id"].nunique(),
        "contracts": {key: len(value) for key, value in contracts.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

