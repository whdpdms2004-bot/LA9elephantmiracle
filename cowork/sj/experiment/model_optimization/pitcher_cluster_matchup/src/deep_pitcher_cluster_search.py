from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))

from screen_reverse_batter_clusters import (  # noqa: E402
    FEATURES as REVERSE_FEATURES,
    add_context_residual,
    build_batter_profile,
    cluster_batters,
    load_main,
)


MODEL_NAME = "xgboost_insight_insight_success_adjusted"
SUCCESS_CONFIG = "m_com_gmm_kme_b3r4_l1000_h1_d72788b5"
SUCCESS_FEATURES = [
    "match_pair_delta", "match_pair_delta_reliability",
    "match_pair_delta_rate", "match_pair_known",
]
CUTOFFS = [2022, 2023, 2024]
SEED = 17
BATTER_K = (4, 6)
SMOOTHING = 1000.0
HALF_LIFE = 1.0
RIDGE_ALPHAS = [100.0, 1000.0, 10000.0]
SUCCESS_SCALES = [0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50]
REVERSE_SCALES = [
    0.0, 0.05, 0.10, 0.15, 0.20,
    0.25, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80,
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="cluster_registry_stage1.csv")
    parser.add_argument("--output-prefix", default="deep_pitcher_cluster")
    parser.add_argument("--cluster-dir", default="clusters")
    parser.add_argument("--r-aware", action="store_true")
    return parser.parse_args()


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
    base = pd.concat(pieces, ignore_index=True)
    success = pd.read_parquet(
        WORK / "oof" / f"matchup_features_{SUCCESS_CONFIG}.parquet",
        columns=["row_id", "season", *SUCCESS_FEATURES],
    )
    game_type = pd.read_csv(
        ROOT / "data" / "train.csv", usecols=["row_id", "game_type"]
    )
    return (
        base.merge(success, on=["row_id", "season"], validate="one_to_one")
        .merge(game_type, on="row_id", validate="one_to_one")
    )


def attach_pitcher_type(frame, cutoff, pitcher_config, cluster_dir="clusters"):
    lookup = pd.read_parquet(
        WORK / cluster_dir / pitcher_config / f"pitcher_lookup_{cutoff}.parquet",
        columns=["pitcher_id", "cluster_code"],
    ).rename(columns={"cluster_code": "pitcher_type"})
    output = frame.merge(lookup, on="pitcher_id", how="left", validate="many_to_one")
    output["pitcher_type"] = output["pitcher_type"].fillna(
        "H" + output["pitcher_hand"].astype(str) + "_new"
    )
    return output


def build_candidate_features(main, pitcher_config, cluster_dir="clusters"):
    pieces = []
    audits = []
    for cutoff in CUTOFFS:
        typed = attach_pitcher_type(
            main.loc[main["season"].le(cutoff)].copy(), cutoff, pitcher_config,
            cluster_dir,
        )
        past = add_context_residual(typed.loc[typed["season"].lt(cutoff)])
        profile = build_batter_profile(past)
        batter_lookup, cluster_audit = cluster_batters(
            profile, "kmeans", BATTER_K, seed=SEED
        )
        past = past.merge(
            batter_lookup[["batter_id", "batter_hand", "batter_type"]],
            on=["batter_id", "batter_hand"], how="left", validate="many_to_one",
        )
        past["batter_type"] = past["batter_type"].fillna(
            "RBH" + past["batter_hand"].astype(str) + "_new"
        )
        weight = np.power(
            0.5, (cutoff - past["season"].to_numpy("float64")) / HALF_LIFE
        )
        work = past[["pitcher_type", "batter_type"]].copy()
        work["weighted_residual"] = past["reverse_residual"].to_numpy(float) * weight
        work["weighted_reverse"] = past["reverse"].to_numpy(float) * weight
        work["weight"] = weight
        pair = work.groupby(["pitcher_type", "batter_type"], sort=False).agg(
            weighted_residual=("weighted_residual", "sum"),
            weighted_reverse=("weighted_reverse", "sum"),
            effective_n=("weight", "sum"),
        ).reset_index()
        pair["reverse_pair_delta"] = pair["weighted_residual"] / (
            pair["effective_n"] + SMOOTHING
        )
        pair["reverse_pair_delta_reliability"] = pair["effective_n"] / (
            pair["effective_n"] + SMOOTHING
        )
        pair["reverse_pair_rate"] = pair["weighted_reverse"] / pair["effective_n"]

        current = typed.loc[typed["season"].eq(cutoff), [
            "row_id", "season", "pitcher_type", "batter_id", "batter_hand"
        ]].copy()
        current = current.merge(
            batter_lookup[["batter_id", "batter_hand", "batter_type"]],
            on=["batter_id", "batter_hand"], how="left", validate="many_to_one",
        )
        current["batter_type"] = current["batter_type"].fillna(
            "RBH" + current["batter_hand"].astype(str) + "_new"
        )
        current = current.merge(
            pair[["pitcher_type", "batter_type", *REVERSE_FEATURES[:-1]]],
            on=["pitcher_type", "batter_type"], how="left", validate="many_to_one",
        )
        current["reverse_pair_known"] = current["reverse_pair_delta"].notna().astype("float32")
        current["reverse_pair_delta"] = current["reverse_pair_delta"].fillna(0.0)
        current["reverse_pair_delta_reliability"] = current[
            "reverse_pair_delta_reliability"
        ].fillna(0.0)
        for column in REVERSE_FEATURES:
            current[column] = current[column].astype("float32")
        pieces.append(current[["row_id", "season", *REVERSE_FEATURES]])
        audits.append({
            "cutoff": cutoff,
            "pitcher_types": int(typed["pitcher_type"].nunique()),
            "batter_types": int(batter_lookup["batter_type"].nunique()),
            "pair_cells": int(len(pair)),
            "coverage": float(current["reverse_pair_known"].mean()),
            "batter_cluster_audit": cluster_audit,
        })
    return pd.concat(pieces, ignore_index=True), audits


def fit_correction(frame, features, alpha, train_year, valid_year):
    train = frame["season"].eq(train_year)
    valid = frame["season"].eq(valid_year)
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=False),
    )
    residual = frame.loc[train, "control_success"] - frame.loc[train, "prediction"]
    model.fit(frame.loc[train, features], residual)
    return np.clip(model.predict(frame.loc[valid, features]), -0.05, 0.05)


def brier_terms(error, success, reverse):
    return {
        "es": float(np.mean(error * success)),
        "er": float(np.mean(error * reverse)),
        "ss": float(np.mean(success ** 2)),
        "rr": float(np.mean(reverse ** 2)),
        "sr": float(np.mean(success * reverse)),
    }


def brier_delta_from_terms(terms, success_scale, reverse_scale):
    s = float(success_scale)
    r = float(reverse_scale)
    return (
        2.0 * s * terms["es"]
        + 2.0 * r * terms["er"]
        + s * s * terms["ss"]
        + r * r * terms["rr"]
        + 2.0 * s * r * terms["sr"]
    )


def robust_objective(f23, f24, denominators):
    n23 = f23 / denominators[2023]
    n24 = f24 / denominators[2024]
    return 0.30 * n23 + 0.70 * n24 + 0.50 * max(n23, n24, 0.0)


def evaluate_candidate(base, features, pitcher_config, intrinsic, r_aware=False):
    frame = base.merge(features, on=["row_id", "season"], validate="one_to_one")
    fold = {}
    success_correction = {}
    for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
        valid = frame["season"].eq(valid_year)
        y = frame.loc[valid, "control_success"].to_numpy(float)
        base_p = frame.loc[valid, "prediction"].to_numpy(float)
        fold[valid_year] = {
            "row_id": frame.loc[valid, "row_id"].to_numpy(),
            "y": y,
            "base": base_p,
            "error": base_p - y,
            "denominator": float(y.mean() * (1.0 - y.mean())),
            "r_mask": frame.loc[valid, "game_type"].eq("R").to_numpy(),
        }
        success_correction[valid_year] = fit_correction(
            frame, SUCCESS_FEATURES, 10.0, train_year, valid_year
        )
    denominators = {year: value["denominator"] for year, value in fold.items()}
    rows = []
    reverse_corrections = {}
    for alpha in RIDGE_ALPHAS:
        reverse = {
            valid_year: fit_correction(
                frame, REVERSE_FEATURES, alpha, train_year, valid_year
            )
            for train_year, valid_year in [(2022, 2023), (2023, 2024)]
        }
        reverse_corrections[alpha] = reverse
        terms = {
            year: brier_terms(
                fold[year]["error"], success_correction[year], reverse[year]
            )
            for year in [2023, 2024]
        }
        r_terms = {
            year: brier_terms(
                fold[year]["error"][fold[year]["r_mask"]],
                success_correction[year][fold[year]["r_mask"]],
                reverse[year][fold[year]["r_mask"]],
            )
            for year in [2023, 2024]
        }
        r_denominators = {
            year: float(
                fold[year]["y"][fold[year]["r_mask"]].mean()
                * (1.0 - fold[year]["y"][fold[year]["r_mask"]].mean())
            )
            for year in [2023, 2024]
        }
        for success_scale in SUCCESS_SCALES:
            for reverse_scale in REVERSE_SCALES:
                deltas = {
                    year: brier_delta_from_terms(
                        terms[year], success_scale, reverse_scale
                    )
                    for year in [2023, 2024]
                }
                r_deltas = {
                    year: brier_delta_from_terms(
                        r_terms[year], success_scale, reverse_scale
                    )
                    for year in [2023, 2024]
                }
                objective = robust_objective(
                    deltas[2023], deltas[2024], denominators
                )
                if r_aware:
                    r23 = r_deltas[2023] / r_denominators[2023]
                    r24 = r_deltas[2024] / r_denominators[2024]
                    objective += (
                        0.25 * (0.30 * r23 + 0.70 * r24)
                        + 1.00 * max(r23, r24, 0.0)
                    )
                rows.append({
                    "pitcher_config": pitcher_config,
                    **intrinsic,
                    "alpha": alpha,
                    "success_scale": success_scale,
                    "reverse_scale": reverse_scale,
                    "f23_delta_brier": deltas[2023],
                    "f24_delta_brier": deltas[2024],
                    "val2023_r_delta_brier": r_deltas[2023],
                    "val2024_r_delta_brier": r_deltas[2024],
                    "both_improve": deltas[2023] < 0 and deltas[2024] < 0,
                    "r_both_improve": r_deltas[2023] < 0 and r_deltas[2024] < 0,
                    "robust_objective": objective,
                })
    result = pd.DataFrame(rows).sort_values("robust_objective")
    best = result.iloc[0].to_dict()
    corrected = np.clip(
        fold[2024]["base"]
        + best["success_scale"] * success_correction[2024]
        + best["reverse_scale"] * reverse_corrections[best["alpha"]][2024],
        1e-6, 1 - 1e-6,
    )
    performance = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    performance = performance.loc[
        performance["track"].eq("performance") & performance["season"].eq(2024),
        ["row_id", "prediction"],
    ].set_index("row_id")
    perf = performance.loc[fold[2024]["row_id"], "prediction"].to_numpy(float)
    y = fold[2024]["y"]
    blend_rows = []
    for weight in np.round(np.arange(0.50, 0.651, 0.002), 3):
        pred = weight * corrected + (1.0 - weight) * perf
        brier = float(np.mean((pred - y) ** 2))
        bss = max(0.0, 100000.0 * (1.0 - brier / fold[2024]["denominator"]))
        blend_rows.append((brier, float(weight), bss))
    brier, weight, bss = min(blend_rows)
    best.update({
        "outer_insight_weight": weight,
        "blend_brier_2024": brier,
        "blend_bss_2024": bss,
        "r_aware_selection": bool(r_aware),
    })
    return result, best


def main():
    args = parse_args()
    main_frame = load_main()
    base = load_base()
    registry = pd.read_csv(WORK / "reports" / args.registry)
    intrinsic = (
        registry.groupby([
            "config_id", "representation", "algorithm", "pca_dim", "k_left", "k_right"
        ], as_index=False)
        .agg(
            min_cluster_size=("min_cluster_size", "min"),
            silhouette=("silhouette", "mean"),
            seed_ari_mean=("seed_ari_mean", "mean"),
            seed_ari_min=("seed_ari_min", "min"),
        )
    )
    all_rows = []
    best_rows = []
    audit_rows = []
    for index, spec in intrinsic.iterrows():
        config = spec["config_id"]
        features, audits = build_candidate_features(
            main_frame, config, args.cluster_dir
        )
        intrinsic_values = {
            key: spec[key]
            for key in [
                "representation", "algorithm", "pca_dim", "k_left", "k_right",
                "min_cluster_size", "silhouette", "seed_ari_mean", "seed_ari_min",
            ]
        }
        result, best = evaluate_candidate(
            base, features, config, intrinsic_values, args.r_aware
        )
        all_rows.append(result)
        best_rows.append(best)
        audit_rows.extend({"pitcher_config": config, **item} for item in audits)
        print(json.dumps({
            "completed": int(index + 1),
            "total": int(len(intrinsic)),
            "pitcher_config": config,
            "robust_objective": best["robust_objective"],
            "f23_delta_brier": best["f23_delta_brier"],
            "f24_delta_brier": best["f24_delta_brier"],
            "blend_bss_2024": best["blend_bss_2024"],
        }, ensure_ascii=False), flush=True)
    reports = WORK / "reports"
    full = pd.concat(all_rows, ignore_index=True).sort_values("robust_objective")
    best = pd.DataFrame(best_rows).sort_values("robust_objective")
    full.to_csv(reports / f"{args.output_prefix}_grid.csv", index=False)
    best.to_csv(reports / f"{args.output_prefix}_best.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(
        reports / f"{args.output_prefix}_audit.csv", index=False
    )
    print("\nTOP 20")
    print(best.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
