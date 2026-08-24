from __future__ import annotations

import argparse
import json
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
MODEL_NAME = "xgboost_insight_insight_success_adjusted"
SUCCESS_CONFIG = "m_com_gmm_kme_b3r4_l1000_h1_d72788b5"
REVERSE_CONFIG = "reverse_context_l2000_h1_0aadebb0"
SUCCESS_FEATURES = [
    "match_pair_delta", "match_pair_delta_reliability",
    "match_pair_delta_rate", "match_pair_known",
]
REVERSE_FEATURES = [
    "reverse_pair_delta", "reverse_pair_delta_reliability",
    "reverse_pair_rate", "reverse_pair_known",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse-config", default=REVERSE_CONFIG)
    parser.add_argument("--reverse-cache-dir", default="reverse")
    parser.add_argument("--reverse-alpha", type=float, default=1000.0)
    parser.add_argument("--output-prefix", default="dual_matchup")
    return parser.parse_args()


def load_frame(reverse_config, reverse_cache_dir):
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
    reverse = pd.read_parquet(
        WORK / "oof" / reverse_cache_dir / f"{reverse_config}.parquet"
    )
    return (
        base.merge(success, on=["row_id", "season"], validate="one_to_one")
        .merge(reverse, on=["row_id", "season"], validate="one_to_one")
    )


def fit_correction(frame, features, alpha, train_year, valid_year):
    train = frame["season"].eq(train_year)
    valid = frame["season"].eq(valid_year)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=False),
    )
    residual = (
        frame.loc[train, "control_success"] - frame.loc[train, "prediction"]
    ).to_numpy("float64")
    model.fit(frame.loc[train, features], residual)
    return np.clip(model.predict(frame.loc[valid, features]), -0.05, 0.05)


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def bss(y, p):
    y = np.asarray(y, dtype="float64")
    return max(0.0, 100000.0 * (1.0 - brier(y, p) / (y.mean() * (1.0 - y.mean()))))


def main():
    args = parse_args()
    frame = load_frame(args.reverse_config, args.reverse_cache_dir)
    folds = {}
    for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
        valid = frame["season"].eq(valid_year)
        folds[valid_year] = {
            "row_id": frame.loc[valid, "row_id"].to_numpy(),
            "y": frame.loc[valid, "control_success"].to_numpy("float64"),
            "base": frame.loc[valid, "prediction"].to_numpy("float64"),
            "success": fit_correction(
                frame, SUCCESS_FEATURES, 10.0, train_year, valid_year
            ),
            "reverse": fit_correction(
                frame, REVERSE_FEATURES, args.reverse_alpha, train_year, valid_year
            ),
        }

    rows = []
    grid = np.round(np.arange(0.0, 1.201, 0.05), 2)
    for success_scale in grid:
        for reverse_scale in grid:
            item = {
                "success_scale": float(success_scale),
                "reverse_scale": float(reverse_scale),
            }
            normalized = []
            for year in [2023, 2024]:
                fold = folds[year]
                pred = np.clip(
                    fold["base"]
                    + success_scale * fold["success"]
                    + reverse_scale * fold["reverse"],
                    1e-6,
                    1 - 1e-6,
                )
                base_br = brier(fold["y"], fold["base"])
                candidate_br = brier(fold["y"], pred)
                norm = (candidate_br - base_br) / (
                    fold["y"].mean() * (1.0 - fold["y"].mean())
                )
                normalized.append(norm)
                item.update({
                    f"f{str(year)[-2:]}_delta_brier": candidate_br - base_br,
                    f"f{str(year)[-2:]}_bss": bss(fold["y"], pred),
                    f"f{str(year)[-2:]}_normalized_delta": norm,
                })
            item["both_improve"] = normalized[0] < 0 and normalized[1] < 0
            item["robust_objective"] = (
                0.30 * normalized[0]
                + 0.70 * normalized[1]
                + 0.50 * max(normalized[0], normalized[1], 0.0)
            )
            rows.append(item)
    result = pd.DataFrame(rows).sort_values("robust_objective")
    selected = result.iloc[0]

    fold = folds[2024]
    corrected = np.clip(
        fold["base"]
        + float(selected["success_scale"]) * fold["success"]
        + float(selected["reverse_scale"]) * fold["reverse"],
        1e-6,
        1 - 1e-6,
    )
    performance = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    performance = performance.loc[
        performance["track"].eq("performance") & performance["season"].eq(2024),
        ["row_id", "prediction"],
    ].set_index("row_id")
    perf = performance.loc[fold["row_id"], "prediction"].to_numpy("float64")
    blend_rows = []
    for insight_weight in np.round(np.arange(0.45, 0.651, 0.001), 3):
        pred = insight_weight * corrected + (1.0 - insight_weight) * perf
        blend_rows.append({
            "insight_weight": float(insight_weight),
            "brier": brier(fold["y"], pred),
            "bss": bss(fold["y"], pred),
        })
    blend = pd.DataFrame(blend_rows).sort_values("brier")

    reports = WORK / "reports"
    result.to_csv(reports / f"{args.output_prefix}_scale_tuning.csv", index=False)
    blend.to_csv(reports / f"{args.output_prefix}_outer_blend_tuning.csv", index=False)
    summary = {
        "success_config": SUCCESS_CONFIG,
        "success_alpha": 10.0,
        "reverse_config": args.reverse_config,
        "reverse_alpha": args.reverse_alpha,
        "selected_success_scale": float(selected["success_scale"]),
        "selected_reverse_scale": float(selected["reverse_scale"]),
        "f23_delta_brier": float(selected["f23_delta_brier"]),
        "f24_delta_brier": float(selected["f24_delta_brier"]),
        "single_bss_2024": float(selected["f24_bss"]),
        "selected_insight_weight": float(blend.iloc[0]["insight_weight"]),
        "blend_bss_2024": float(blend.iloc[0]["bss"]),
        "scale_top10": result.head(10).to_dict(orient="records"),
        "blend_top5": blend.head(5).to_dict(orient="records"),
    }
    (reports / f"{args.output_prefix}_tuning.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
