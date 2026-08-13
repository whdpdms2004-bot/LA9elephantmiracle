from __future__ import annotations

import itertools
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
    FEATURES,
    add_context_residual,
    add_pitcher_type,
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
SEEDS = [17, 43, 97, 2026, 4099]
CUTOFFS = [2022, 2023, 2024]
ALPHAS = [1000.0, 10000.0, 100000.0]
SMOOTHING = 1000.0
HALF_LIFE = 1.0


def build_seed_features(main):
    pieces = {seed: [] for seed in SEEDS}
    audits = []
    for cutoff in CUTOFFS:
        typed = add_pitcher_type(main.loc[main["season"].le(cutoff)].copy(), cutoff)
        past = add_context_residual(typed.loc[typed["season"].lt(cutoff)])
        profile = build_batter_profile(past)
        current_base = typed.loc[typed["season"].eq(cutoff), [
            "row_id", "season", "pitcher_type", "batter_id", "batter_hand"
        ]].copy()
        for seed in SEEDS:
            lookup, cluster_audit = cluster_batters(
                profile, "kmeans", (4, 6), seed=seed
            )
            typed_past = past.merge(
                lookup[["batter_id", "batter_hand", "batter_type"]],
                on=["batter_id", "batter_hand"], how="left", validate="many_to_one",
            )
            typed_past["batter_type"] = typed_past["batter_type"].fillna(
                "RBH" + typed_past["batter_hand"].astype(str) + "_new"
            )
            current = current_base.merge(
                lookup[["batter_id", "batter_hand", "batter_type"]],
                on=["batter_id", "batter_hand"], how="left", validate="many_to_one",
            )
            current["batter_type"] = current["batter_type"].fillna(
                "RBH" + current["batter_hand"].astype(str) + "_new"
            )
            weight = np.power(
                0.5,
                (cutoff - typed_past["season"].to_numpy("float64")) / HALF_LIFE,
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
            pair["reverse_pair_delta"] = pair["weighted_residual"] / (
                pair["effective_n"] + SMOOTHING
            )
            pair["reverse_pair_delta_reliability"] = pair["effective_n"] / (
                pair["effective_n"] + SMOOTHING
            )
            pair["reverse_pair_rate"] = pair["weighted_reverse"] / pair["effective_n"]
            out = current.merge(
                pair[["pitcher_type", "batter_type", *FEATURES[:-1]]],
                on=["pitcher_type", "batter_type"], how="left", validate="many_to_one",
            )
            out["reverse_pair_known"] = out["reverse_pair_delta"].notna().astype("float32")
            out["reverse_pair_delta"] = out["reverse_pair_delta"].fillna(0.0)
            out["reverse_pair_delta_reliability"] = out[
                "reverse_pair_delta_reliability"
            ].fillna(0.0)
            for column in FEATURES:
                out[column] = out[column].astype("float32")
            pieces[seed].append(out[["row_id", "season", *FEATURES]])
            audits.append({
                "cutoff": cutoff,
                "seed": seed,
                "pair_cells": int(len(pair)),
                "coverage": float(out["reverse_pair_known"].mean()),
                "cluster_audit": json.dumps(cluster_audit),
            })
        print(json.dumps({"built_cutoff": cutoff}, ensure_ascii=False), flush=True)
    cache_dir = WORK / "oof" / "reverse_batter_seed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for seed, seed_pieces in pieces.items():
        pd.concat(seed_pieces, ignore_index=True).to_parquet(
            cache_dir / f"seed_{seed}.parquet", index=False
        )
    pd.DataFrame(audits).to_csv(
        WORK / "reports" / "reverse_batter_seed_audit.csv", index=False
    )


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
    return base.merge(success, on=["row_id", "season"], validate="one_to_one")


def correction(frame, features, alpha, train_year, valid_year):
    train = frame["season"].eq(train_year)
    valid = frame["season"].eq(valid_year)
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=False),
    )
    residual = frame.loc[train, "control_success"] - frame.loc[train, "prediction"]
    model.fit(frame.loc[train, features], residual)
    return np.clip(model.predict(frame.loc[valid, features]), -0.05, 0.05)


def robust_objective(f23_delta, f24_delta, fold_data):
    n23 = f23_delta / fold_data[2023]["denominator"]
    n24 = f24_delta / fold_data[2024]["denominator"]
    return 0.30 * n23 + 0.70 * n24 + 0.50 * max(n23, n24, 0.0)


def brier_delta(error, correction_value):
    return float(np.mean(2.0 * error * correction_value + correction_value ** 2))


def main():
    build_seed_features(load_main())
    base = load_base()
    seed_frames = {}
    cache_dir = WORK / "oof" / "reverse_batter_seed"
    for seed in SEEDS:
        seed_frames[seed] = base.merge(
            pd.read_parquet(cache_dir / f"seed_{seed}.parquet"),
            on=["row_id", "season"], validate="one_to_one",
        )
    fold_data = {}
    success_corrections = {}
    for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
        valid = base["season"].eq(valid_year)
        y = base.loc[valid, "control_success"].to_numpy(float)
        base_p = base.loc[valid, "prediction"].to_numpy(float)
        fold_data[valid_year] = {
            "row_id": base.loc[valid, "row_id"].to_numpy(),
            "y": y,
            "base": base_p,
            "error": base_p - y,
            "denominator": float(y.mean() * (1.0 - y.mean())),
        }
        success_corrections[valid_year] = correction(
            base, SUCCESS_FEATURES, 10.0, train_year, valid_year
        )

    reverse_corrections = {}
    for alpha in ALPHAS:
        for seed in SEEDS:
            reverse_corrections[(alpha, seed)] = {}
            for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
                reverse_corrections[(alpha, seed)][valid_year] = correction(
                    seed_frames[seed], FEATURES, alpha, train_year, valid_year
                )

    subset_rows = []
    subset_predictions = {}
    for alpha in ALPHAS:
        for size in range(1, len(SEEDS) + 1):
            for subset in itertools.combinations(SEEDS, size):
                key = f"a{int(alpha)}_" + "-".join(map(str, subset))
                subset_predictions[key] = {}
                item = {"key": key, "alpha": alpha, "seeds": "-".join(map(str, subset)), "seed_count": size}
                best_scales = {}
                deltas = {}
                for valid_year in [2023, 2024]:
                    mean_correction = np.mean(
                        [reverse_corrections[(alpha, seed)][valid_year] for seed in subset], axis=0
                    )
                    subset_predictions[key][valid_year] = mean_correction
                    error = fold_data[valid_year]["error"]
                    candidates = []
                    for scale in np.round(np.arange(0.0, 1.201, 0.05), 2):
                        delta = brier_delta(error, scale * mean_correction)
                        candidates.append((delta, float(scale)))
                    best_delta, best_scale = min(candidates)
                    best_scales[valid_year] = best_scale
                    deltas[valid_year] = best_delta
                    item[f"f{str(valid_year)[-2:]}_best_scale"] = best_scale
                    item[f"f{str(valid_year)[-2:]}_best_delta"] = best_delta
                common = []
                for scale in np.round(np.arange(0.0, 1.201, 0.05), 2):
                    f23 = brier_delta(
                        fold_data[2023]["error"], scale * subset_predictions[key][2023]
                    )
                    f24 = brier_delta(
                        fold_data[2024]["error"], scale * subset_predictions[key][2024]
                    )
                    common.append((robust_objective(f23, f24, fold_data), scale, f23, f24))
                objective, scale, f23, f24 = min(common)
                item.update({
                    "common_scale": float(scale),
                    "f23_delta_brier": f23,
                    "f24_delta_brier": f24,
                    "both_improve": f23 < 0 and f24 < 0,
                    "robust_objective": objective,
                })
                subset_rows.append(item)
    subset_result = pd.DataFrame(subset_rows).sort_values("robust_objective")

    joint_rows = []
    for _, candidate in subset_result.head(12).iterrows():
        key = candidate["key"]
        for success_scale in np.round(np.arange(0.0, 0.801, 0.05), 2):
            for reverse_scale in np.round(np.arange(0.0, 1.001, 0.05), 2):
                deltas = {}
                for year in [2023, 2024]:
                    value = (
                        success_scale * success_corrections[year]
                        + reverse_scale * subset_predictions[key][year]
                    )
                    deltas[year] = brier_delta(fold_data[year]["error"], value)
                joint_rows.append({
                    "key": key,
                    "alpha": candidate["alpha"],
                    "seeds": candidate["seeds"],
                    "seed_count": candidate["seed_count"],
                    "success_scale": float(success_scale),
                    "reverse_scale": float(reverse_scale),
                    "f23_delta_brier": deltas[2023],
                    "f24_delta_brier": deltas[2024],
                    "both_improve": deltas[2023] < 0 and deltas[2024] < 0,
                    "robust_objective": robust_objective(
                        deltas[2023], deltas[2024], fold_data
                    ),
                })
    joint_result = pd.DataFrame(joint_rows).sort_values("robust_objective")
    selected = joint_result.iloc[0]
    corrected_2024 = np.clip(
        fold_data[2024]["base"]
        + float(selected["success_scale"]) * success_corrections[2024]
        + float(selected["reverse_scale"]) * subset_predictions[selected["key"]][2024],
        1e-6, 1 - 1e-6,
    )
    performance = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    performance = performance.loc[
        performance["track"].eq("performance") & performance["season"].eq(2024),
        ["row_id", "prediction"],
    ].set_index("row_id")
    perf = performance.loc[fold_data[2024]["row_id"], "prediction"].to_numpy(float)
    y = fold_data[2024]["y"]
    blend_rows = []
    for weight in np.round(np.arange(0.45, 0.651, 0.001), 3):
        pred = weight * corrected_2024 + (1.0 - weight) * perf
        brier = float(np.mean((pred - y) ** 2))
        bss = max(0.0, 100000.0 * (1.0 - brier / fold_data[2024]["denominator"]))
        blend_rows.append({"insight_weight": weight, "brier": brier, "bss": bss})
    blend = pd.DataFrame(blend_rows).sort_values("brier")

    reports = WORK / "reports"
    subset_result.to_csv(reports / "reverse_batter_seed_screen.csv", index=False)
    joint_result.to_csv(reports / "reverse_batter_seed_joint_tuning.csv", index=False)
    blend.to_csv(reports / "reverse_batter_seed_blend_tuning.csv", index=False)
    summary = {
        "selected": selected.to_dict(),
        "blend": blend.iloc[0].to_dict(),
        "top_subsets": subset_result.head(10).to_dict(orient="records"),
        "top_joint": joint_result.head(10).to_dict(orient="records"),
    }
    (reports / "reverse_batter_seed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
