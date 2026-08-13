from __future__ import annotations

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
CONFIG = "m_com_gmm_kme_b3r4_l1000_h1_d72788b5"
FEATURES = [
    "match_pair_delta",
    "match_pair_delta_reliability",
    "match_pair_delta_rate",
    "match_pair_known",
]


def load_base() -> pd.DataFrame:
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


def fit_correction(frame: pd.DataFrame, train_year: int, valid_year: int):
    train = frame["season"].eq(train_year)
    valid = frame["season"].eq(valid_year)
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=10.0, fit_intercept=False),
    )
    residual = (
        frame.loc[train, "control_success"] - frame.loc[train, "prediction"]
    ).to_numpy("float64")
    pipeline.fit(frame.loc[train, FEATURES], residual)
    correction = np.clip(pipeline.predict(frame.loc[valid, FEATURES]), -0.05, 0.05)
    return frame.loc[valid].copy(), correction


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def bss(y, p) -> float:
    y = np.asarray(y, dtype="float64")
    return max(0.0, 100000.0 * (1.0 - brier(y, p) / (y.mean() * (1.0 - y.mean()))))


def main():
    base = load_base()
    features = pd.read_parquet(WORK / "oof" / f"matchup_features_{CONFIG}.parquet")
    frame = base.merge(features[["row_id", "season", *FEATURES]], on=["row_id", "season"], validate="one_to_one")

    folds = {}
    for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
        valid, correction = fit_correction(frame, train_year, valid_year)
        folds[valid_year] = {
            "valid": valid,
            "correction": correction,
        }

    scale_rows = []
    for scale in np.round(np.arange(0.0, 1.501, 0.05), 2):
        item = {"correction_scale": float(scale)}
        normalized_deltas = []
        for valid_year in [2023, 2024]:
            valid = folds[valid_year]["valid"]
            correction = folds[valid_year]["correction"]
            y = valid["control_success"].to_numpy("float64")
            base_p = valid["prediction"].to_numpy("float64")
            pred = np.clip(base_p + scale * correction, 1e-6, 1 - 1e-6)
            base_br = brier(y, base_p)
            br = brier(y, pred)
            denom = y.mean() * (1.0 - y.mean())
            normalized_delta = (br - base_br) / denom
            normalized_deltas.append(normalized_delta)
            item.update({
                f"f{str(valid_year)[-2:]}_brier": br,
                f"f{str(valid_year)[-2:]}_delta_brier": br - base_br,
                f"f{str(valid_year)[-2:]}_bss": bss(y, pred),
                f"f{str(valid_year)[-2:]}_normalized_delta": normalized_delta,
            })
        item["robust_objective"] = (
            0.30 * normalized_deltas[0]
            + 0.70 * normalized_deltas[1]
            + 0.50 * max(normalized_deltas[0], normalized_deltas[1], 0.0)
        )
        item["both_improve"] = normalized_deltas[0] < 0 and normalized_deltas[1] < 0
        scale_rows.append(item)
    scale_result = pd.DataFrame(scale_rows).sort_values("robust_objective")
    selected_scale = float(scale_result.iloc[0]["correction_scale"])

    valid = folds[2024]["valid"]
    correction = folds[2024]["correction"]
    performance = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    performance = performance.loc[
        performance["track"].eq("performance") & performance["season"].eq(2024),
        ["row_id", "prediction"],
    ].rename(columns={"prediction": "performance_prediction"})
    valid = valid.merge(performance, on="row_id", validate="one_to_one")
    corrected = np.clip(
        valid["prediction"].to_numpy("float64") + selected_scale * correction,
        1e-6,
        1 - 1e-6,
    )
    perf = valid["performance_prediction"].to_numpy("float64")
    y = valid["control_success"].to_numpy("float64")
    blend_rows = []
    for weight in np.round(np.arange(0.45, 0.611, 0.002), 3):
        pred = weight * corrected + (1.0 - weight) * perf
        blend_rows.append({
            "correction_scale": selected_scale,
            "insight_weight": float(weight),
            "performance_weight": float(1.0 - weight),
            "brier": brier(y, pred),
            "bss": bss(y, pred),
        })
    blend_result = pd.DataFrame(blend_rows).sort_values("brier")

    report_dir = WORK / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    scale_result.to_csv(report_dir / "matchup_correction_scale_tuning.csv", index=False)
    blend_result.to_csv(report_dir / "matchup_outer_blend_tuning.csv", index=False)
    summary = {
        "config": CONFIG,
        "features": FEATURES,
        "alpha": 10.0,
        "selected_correction_scale": selected_scale,
        "scale_selection": "30% F23 + 70% F24 normalized Brier delta; penalty for a worsening fold",
        "selected_insight_weight": float(blend_result.iloc[0]["insight_weight"]),
        "selected_2024_bss": float(blend_result.iloc[0]["bss"]),
        "selected_2024_brier": float(blend_result.iloc[0]["brier"]),
        "scale_top5": scale_result.head(5).to_dict(orient="records"),
        "blend_top5": blend_result.head(5).to_dict(orient="records"),
    }
    (report_dir / "matchup_submission_tuning.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
