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
FINAL_DIR = WORK / "final" / "robust_matchup_v1"
MODEL_NAME = "xgboost_insight_insight_success_adjusted"
CUTOFF = 2025
SMOOTHING = 2000.0
HALF_LIFE = 1.0
RIDGE_ALPHA = 1000.0
FEATURES = [
    "reverse_pair_delta",
    "reverse_pair_delta_reliability",
    "reverse_pair_rate",
    "reverse_pair_known",
]


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


def build_pair_table(main):
    pitcher = pd.read_csv(
        FINAL_DIR / "pitcher_lookup_2025.csv",
        usecols=["pitcher_id", "pitcher_type"],
    )
    batter = pd.read_csv(
        FINAL_DIR / "batter_lookup_2025.csv",
        usecols=["batter_id", "batter_hand", "batter_type"],
    )
    past = main.loc[main["reverse"].notna()].copy()
    past = past.merge(pitcher, on="pitcher_id", how="left", validate="many_to_one")
    past["pitcher_type"] = past["pitcher_type"].fillna(
        "H" + past["pitcher_hand"].astype(str) + "_new"
    )
    past = past.merge(
        batter,
        on=["batter_id", "batter_hand"],
        how="left",
        validate="many_to_one",
    )
    past["batter_type"] = past["batter_type"].fillna(
        "BH" + past["batter_hand"].astype(str) + "_new"
    )
    context_keys = [
        "season", "pitcher_hand", "batter_hand", "balls_before", "strikes_before"
    ]
    expected = past.groupby(context_keys)["reverse"].transform("mean")
    season_rate = past.groupby("season")["reverse"].transform("mean")
    past["reverse_residual"] = past["reverse"] - expected.fillna(season_rate)
    past["recency_weight"] = np.power(
        0.5, (CUTOFF - past["season"].to_numpy("float64")) / HALF_LIFE
    )
    past["weighted_residual"] = past["reverse_residual"] * past["recency_weight"]
    past["weighted_reverse"] = past["reverse"] * past["recency_weight"]
    pair = past.groupby(["pitcher_type", "batter_type"], sort=False).agg(
        weighted_residual=("weighted_residual", "sum"),
        weighted_reverse=("weighted_reverse", "sum"),
        effective_n=("recency_weight", "sum"),
        raw_n=("recency_weight", "size"),
    ).reset_index()
    pair["reverse_pair_delta"] = pair["weighted_residual"] / (
        pair["effective_n"] + SMOOTHING
    )
    pair["reverse_pair_delta_reliability"] = pair["effective_n"] / (
        pair["effective_n"] + SMOOTHING
    )
    pair["reverse_pair_rate"] = pair["weighted_reverse"] / pair["effective_n"]
    return pair[[
        "pitcher_type", "batter_type", "effective_n", "raw_n",
        "reverse_pair_delta", "reverse_pair_delta_reliability", "reverse_pair_rate",
    ]], past


def fit_ridge():
    prediction = pd.read_parquet(
        MODEL_DIR / "insight_feature_ablation_predictions_success_screen_2024.parquet"
    )
    prediction = prediction.loc[prediction["model"].eq(MODEL_NAME)]
    features = pd.read_parquet(
        WORK / "oof" / "reverse" / "reverse_context_l2000_h1_0aadebb0.parquet"
    )
    frame = prediction.merge(features, on=["row_id", "season"], validate="one_to_one")
    residual = frame["control_success"] - frame["prediction"]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA, fit_intercept=False),
    )
    model.fit(frame[FEATURES], residual)
    imputer, scaler, ridge = model
    transformed = (imputer.transform(frame[FEATURES]) - scaler.mean_) / scaler.scale_
    manual = transformed @ ridge.coef_
    library = model.predict(frame[FEATURES])
    if not np.allclose(manual, library, atol=1e-8, rtol=1e-8):
        raise RuntimeError("Manual reverse Ridge export does not reproduce sklearn")
    return {
        "feature_order": FEATURES,
        "imputer_statistics": imputer.statistics_.astype(float).tolist(),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "ridge_coef": ridge.coef_.astype(float).tolist(),
        "ridge_alpha": RIDGE_ALPHA,
        "correction_clip": [-0.05, 0.05],
        "train_season": 2024,
        "rows": int(len(frame)),
        "train_residual_mean": float(residual.mean()),
        "train_correction_mean": float(library.mean()),
        "train_correction_std": float(library.std()),
        "manual_max_abs_diff": float(np.max(np.abs(manual - library))),
    }


def main():
    main_frame = load_main()
    pair, past = build_pair_table(main_frame)
    ridge = fit_ridge()
    pair.to_csv(FINAL_DIR / "reverse_pair_table_2025.csv", index=False)
    pair.to_parquet(FINAL_DIR / "reverse_pair_table_2025.parquet", index=False)
    (FINAL_DIR / "reverse_ridge_correction.json").write_text(
        json.dumps(ridge, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "version": "reverse_matchup_v1",
        "cutoff": CUTOFF,
        "main_trained_through": 2024,
        "center_mode": "season x pitcher_hand x batter_hand x count_state",
        "smoothing": SMOOTHING,
        "half_life": HALF_LIFE,
        "pair_cells": int(len(pair)),
        "valid_history_rows": int(len(past)),
        "ridge": ridge,
    }
    (FINAL_DIR / "reverse_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
