from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
PITCHER_CONFIG = "com_gmm_p8_l2r4_7de7125e"
MODEL_NAME = "xgboost_insight_insight_success_adjusted"
SEED = 2026
CUTOFFS = [2022, 2023, 2024]
ALGORITHMS = ["kmeans", "gmm_diag"]
K_PAIRS = [(2, 3), (3, 4), (4, 6), (6, 8), (8, 12)]
SMOOTHINGS = [1000.0, 2000.0, 4000.0]
HALF_LIVES = [1.0, 2.0]
ALPHAS = [100.0, 1000.0, 10000.0]
PROFILE_LAMBDA = 200.0
FEATURES = [
    "reverse_pair_delta", "reverse_pair_delta_reliability",
    "reverse_pair_rate", "reverse_pair_known",
]


def config_name(algorithm, k_pair, smoothing, half_life):
    raw = f"{algorithm}|{k_pair}|{smoothing}|{half_life}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return (
        f"revbat_{algorithm[:3]}_b{k_pair[0]}r{k_pair[1]}_"
        f"l{int(smoothing)}_h{int(half_life)}_{digest}"
    )


def load_main():
    main = pd.read_csv(
        ROOT / "data" / "train.csv",
        usecols=[
            "row_id", "season", "pitcher_id", "pitcher_hand", "batter_id",
            "batter_hand", "balls_before", "strikes_before",
        ],
    )
    labels = pd.read_parquet(
        MODEL_DIR / "failure_component_labels.parquet",
        columns=["row_id", "reverse"],
    )
    if not main["row_id"].equals(labels["row_id"]):
        raise RuntimeError("Failure labels are not row-aligned")
    main["reverse"] = labels["reverse"].astype("float32")
    return main


def add_pitcher_type(frame, cutoff):
    lookup = pd.read_parquet(
        WORK / "clusters" / PITCHER_CONFIG / f"pitcher_lookup_{cutoff}.parquet",
        columns=["pitcher_id", "cluster_code"],
    ).rename(columns={"cluster_code": "pitcher_type"})
    output = frame.merge(lookup, on="pitcher_id", how="left", validate="many_to_one")
    output["pitcher_type"] = output["pitcher_type"].fillna(
        "H" + output["pitcher_hand"].astype(str) + "_new"
    )
    return output


def add_context_residual(past):
    output = past.loc[past["reverse"].notna()].copy()
    keys = ["season", "pitcher_hand", "batter_hand", "balls_before", "strikes_before"]
    expected = output.groupby(keys)["reverse"].transform("mean")
    season_rate = output.groupby("season")["reverse"].transform("mean")
    output["reverse_residual"] = output["reverse"] - expected.fillna(season_rate)
    return output


def build_batter_profile(past):
    base = past.groupby(["batter_id", "batter_hand"], sort=False).agg(
        batter_n=("reverse", "size"),
        resid_sum=("reverse_residual", "sum"),
    ).reset_index()
    base["batter_reverse_resid"] = base["resid_sum"] / (
        base["batter_n"] + PROFILE_LAMBDA
    )
    for key, prefix in [("pitcher_type", "vs_pt"), ("pitcher_hand", "vs_ph")]:
        group = past.groupby(["batter_id", "batter_hand", key], sort=False).agg(
            n=("reverse", "size"), resid_sum=("reverse_residual", "sum")
        ).reset_index()
        group["value"] = group["resid_sum"] / (group["n"] + PROFILE_LAMBDA)
        value = group.pivot(index=["batter_id", "batter_hand"], columns=key, values="value")
        value.columns = [f"revprof_{prefix}_{item}_resid" for item in value.columns]
        count = group.pivot(index=["batter_id", "batter_hand"], columns=key, values="n")
        count.columns = [f"revprof_{prefix}_{item}_logn" for item in count.columns]
        base = base.merge(value, left_on=["batter_id", "batter_hand"], right_index=True, how="left")
        base = base.merge(np.log1p(count), left_on=["batter_id", "batter_hand"], right_index=True, how="left")
    return base


def cluster_batters(profile, algorithm, k_pair, seed=SEED):
    feature_columns = [
        column for column in profile
        if column.startswith("batter_reverse_") or column.startswith("revprof_")
    ]
    pieces = []
    audits = []
    for hand, k in [(1, k_pair[0]), (2, k_pair[1])]:
        part = profile.loc[profile["batter_hand"].eq(hand)].copy()
        raw = part[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy("float64")
        values = SimpleImputer(strategy="median").fit_transform(raw)
        values = RobustScaler(quantile_range=(10, 90)).fit_transform(values)
        values = np.clip(values, -5.0, 5.0)
        dim = max(1, min(8, values.shape[1], len(part) - 1))
        embedding = PCA(n_components=dim, random_state=seed).fit_transform(values)
        if algorithm == "kmeans":
            labels = KMeans(n_clusters=k, n_init=30, random_state=seed).fit_predict(embedding)
        else:
            labels = GaussianMixture(
                n_components=k, covariance_type="diag", reg_covar=1e-4,
                n_init=5, max_iter=500, random_state=seed,
            ).fit_predict(embedding)
        overall = part["batter_reverse_resid"].to_numpy("float64")
        ordering = sorted(range(k), key=lambda value: float(np.nanmedian(overall[labels == value])))
        mapping = {old: new for new, old in enumerate(ordering)}
        labels = np.asarray([mapping[int(value)] for value in labels], dtype="int16")
        result = part[["batter_id", "batter_hand", "batter_n", "batter_reverse_resid"]].copy()
        result["batter_cluster"] = labels
        result["batter_type"] = [f"RBH{hand}_C{int(value):02d}" for value in labels]
        pieces.append(result)
        counts = np.bincount(labels, minlength=k)
        audits.append({
            "hand": hand,
            "k": k,
            "batters": int(len(part)),
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
        })
    return pd.concat(pieces, ignore_index=True), audits


def build_all_features(main):
    configs = {
        config_name(algorithm, k_pair, smoothing, half_life): []
        for algorithm in ALGORITHMS
        for k_pair in K_PAIRS
        for smoothing in SMOOTHINGS
        for half_life in HALF_LIVES
    }
    audits = []
    for cutoff in CUTOFFS:
        typed = add_pitcher_type(main.loc[main["season"].le(cutoff)].copy(), cutoff)
        past = add_context_residual(typed.loc[typed["season"].lt(cutoff)])
        profile = build_batter_profile(past)
        current_base = typed.loc[typed["season"].eq(cutoff), [
            "row_id", "season", "pitcher_type", "batter_id", "batter_hand"
        ]].copy()
        for algorithm in ALGORITHMS:
            for k_pair in K_PAIRS:
                batter_lookup, cluster_audit = cluster_batters(profile, algorithm, k_pair)
                typed_past = past.merge(
                    batter_lookup[["batter_id", "batter_hand", "batter_type"]],
                    on=["batter_id", "batter_hand"], how="left", validate="many_to_one",
                )
                typed_past["batter_type"] = typed_past["batter_type"].fillna(
                    "RBH" + typed_past["batter_hand"].astype(str) + "_new"
                )
                current = current_base.merge(
                    batter_lookup[["batter_id", "batter_hand", "batter_type"]],
                    on=["batter_id", "batter_hand"], how="left", validate="many_to_one",
                )
                current["batter_type"] = current["batter_type"].fillna(
                    "RBH" + current["batter_hand"].astype(str) + "_new"
                )
                for half_life in HALF_LIVES:
                    weight = np.power(
                        0.5,
                        (cutoff - typed_past["season"].to_numpy("float64")) / half_life,
                    )
                    work = typed_past[["pitcher_type", "batter_type"]].copy()
                    work["weighted_residual"] = typed_past["reverse_residual"].to_numpy(float) * weight
                    work["weighted_reverse"] = typed_past["reverse"].to_numpy(float) * weight
                    work["weight"] = weight
                    pair = work.groupby(["pitcher_type", "batter_type"], sort=False).agg(
                        weighted_residual=("weighted_residual", "sum"),
                        weighted_reverse=("weighted_reverse", "sum"),
                        effective_n=("weight", "sum"),
                    ).reset_index()
                    for smoothing in SMOOTHINGS:
                        config = config_name(algorithm, k_pair, smoothing, half_life)
                        table = pair[[
                            "pitcher_type", "batter_type", "weighted_residual",
                            "weighted_reverse", "effective_n",
                        ]].copy()
                        table["reverse_pair_delta"] = table["weighted_residual"] / (
                            table["effective_n"] + smoothing
                        )
                        table["reverse_pair_delta_reliability"] = table["effective_n"] / (
                            table["effective_n"] + smoothing
                        )
                        table["reverse_pair_rate"] = table["weighted_reverse"] / table["effective_n"]
                        out = current.merge(
                            table[["pitcher_type", "batter_type", *FEATURES[:-1]]],
                            on=["pitcher_type", "batter_type"], how="left", validate="many_to_one",
                        )
                        out["reverse_pair_known"] = out["reverse_pair_delta"].notna().astype("float32")
                        out["reverse_pair_delta"] = out["reverse_pair_delta"].fillna(0.0)
                        out["reverse_pair_delta_reliability"] = out[
                            "reverse_pair_delta_reliability"
                        ].fillna(0.0)
                        for column in FEATURES:
                            out[column] = out[column].astype("float32")
                        configs[config].append(out[["row_id", "season", *FEATURES]])
                        audits.append({
                            "config": config,
                            "cutoff": cutoff,
                            "algorithm": algorithm,
                            "k_left": k_pair[0],
                            "k_right": k_pair[1],
                            "smoothing": smoothing,
                            "half_life": half_life,
                            "pair_cells": int(len(table)),
                            "coverage": float(out["reverse_pair_known"].mean()),
                            "cluster_audit": json.dumps(cluster_audit),
                        })
        print(json.dumps({"built_cutoff": cutoff}, ensure_ascii=False), flush=True)
    cache_dir = WORK / "oof" / "reverse_batter"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for config, pieces in configs.items():
        pd.concat(pieces, ignore_index=True).to_parquet(cache_dir / f"{config}.parquet", index=False)
    pd.DataFrame(audits).to_csv(WORK / "reports" / "reverse_batter_cluster_audit.csv", index=False)
    return list(configs)


def load_base():
    paths = [
        MODEL_DIR / "insight_feature_ablation_predictions_cluster_base_2022.parquet",
        MODEL_DIR / "insight_feature_ablation_predictions_success_adjusted_2023.parquet",
        MODEL_DIR / "insight_feature_ablation_predictions_success_screen_2024.parquet",
    ]
    pieces = []
    for path in paths:
        frame = pd.read_parquet(path)
        if frame["model"].nunique() > 1:
            frame = frame.loc[frame["model"].eq(MODEL_NAME)]
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def evaluate_fold(frame, alpha, train_year, valid_year):
    train = frame["season"].eq(train_year)
    valid = frame["season"].eq(valid_year)
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=False),
    )
    residual = frame.loc[train, "control_success"] - frame.loc[train, "prediction"]
    model.fit(frame.loc[train, FEATURES], residual)
    correction = np.clip(model.predict(frame.loc[valid, FEATURES]), -0.05, 0.05)
    y = frame.loc[valid, "control_success"].to_numpy(float)
    base = frame.loc[valid, "prediction"].to_numpy(float)
    prediction = np.clip(base + correction, 1e-6, 1 - 1e-6)
    base_brier = float(np.mean((base - y) ** 2))
    candidate_brier = float(np.mean((prediction - y) ** 2))
    return {
        "delta_brier": candidate_brier - base_brier,
        "normalized_delta": (candidate_brier - base_brier) / (y.mean() * (1 - y.mean())),
        "correction_std": float(correction.std()),
    }


def screen(configs):
    base = load_base()
    rows = []
    cache_dir = WORK / "oof" / "reverse_batter"
    for config in configs:
        frame = base.merge(
            pd.read_parquet(cache_dir / f"{config}.parquet"),
            on=["row_id", "season"], validate="one_to_one",
        )
        tokens = config.split("_")
        for alpha in ALPHAS:
            f23 = evaluate_fold(frame, alpha, 2022, 2023)
            f24 = evaluate_fold(frame, alpha, 2023, 2024)
            objective = (
                0.30 * f23["normalized_delta"]
                + 0.70 * f24["normalized_delta"]
                + 0.50 * max(f23["normalized_delta"], f24["normalized_delta"], 0.0)
            )
            rows.append({
                "config": config,
                "algorithm": tokens[1],
                "k_left": int(tokens[2].split("r")[0][1:]),
                "k_right": int(tokens[2].split("r")[1]),
                "smoothing": float(tokens[3][1:]),
                "half_life": float(tokens[4][1:]),
                "alpha": alpha,
                "f23_delta_brier": f23["delta_brier"],
                "f24_delta_brier": f24["delta_brier"],
                "f23_correction_std": f23["correction_std"],
                "f24_correction_std": f24["correction_std"],
                "both_improve": f23["delta_brier"] < 0 and f24["delta_brier"] < 0,
                "robust_objective": objective,
            })
        print(json.dumps({"screened": config}, ensure_ascii=False), flush=True)
    result = pd.DataFrame(rows).sort_values("robust_objective")
    result.to_csv(WORK / "reports" / "reverse_batter_cluster_screen.csv", index=False)
    print("\nTOP 20")
    print(result.head(20).to_string(index=False))
    print(json.dumps({
        "runs": len(result),
        "both_improve": int(result["both_improve"].sum()),
        "best": result.iloc[0].to_dict(),
    }, ensure_ascii=False, indent=2))


def main():
    configs = build_all_features(load_main())
    screen(configs)


if __name__ == "__main__":
    main()
