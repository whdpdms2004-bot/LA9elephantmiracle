from __future__ import annotations

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
FINAL_DIR = WORK / "final" / "robust_matchup_v1"
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


CUTOFF = 2025
BATTER_K = (4, 6)
SMOOTHING = 1000.0
HALF_LIFE = 1.0
RIDGE_ALPHA = 10000.0
OOF_CONFIG = "revbat_kme_b4r6_l1000_h1_dcf9f751"
MODEL_NAME = "xgboost_insight_insight_success_adjusted"


def build_final_tables(main):
    typed = add_pitcher_type(main.copy(), CUTOFF)
    past = add_context_residual(typed)
    profile = build_batter_profile(past)
    batter_lookup, cluster_audit = cluster_batters(profile, "kmeans", BATTER_K)
    past = past.merge(
        batter_lookup[["batter_id", "batter_hand", "batter_type"]],
        on=["batter_id", "batter_hand"],
        how="left",
        validate="many_to_one",
    )
    past["batter_type"] = past["batter_type"].fillna(
        "RBH" + past["batter_hand"].astype(str) + "_new"
    )
    weight = np.power(
        0.5, (CUTOFF - past["season"].to_numpy("float64")) / HALF_LIFE
    )
    work = past[["pitcher_type", "batter_type"]].copy()
    work["weighted_residual"] = past["reverse_residual"].to_numpy(float) * weight
    work["weighted_reverse"] = past["reverse"].to_numpy(float) * weight
    work["weight"] = weight
    pair = work.groupby(["pitcher_type", "batter_type"], sort=False).agg(
        weighted_residual=("weighted_residual", "sum"),
        weighted_reverse=("weighted_reverse", "sum"),
        effective_n=("weight", "sum"),
        raw_n=("weight", "size"),
    ).reset_index()
    pair["reverse_pair_delta"] = pair["weighted_residual"] / (
        pair["effective_n"] + SMOOTHING
    )
    pair["reverse_pair_delta_reliability"] = pair["effective_n"] / (
        pair["effective_n"] + SMOOTHING
    )
    pair["reverse_pair_rate"] = pair["weighted_reverse"] / pair["effective_n"]
    pair = pair[[
        "pitcher_type", "batter_type", "effective_n", "raw_n",
        "reverse_pair_delta", "reverse_pair_delta_reliability", "reverse_pair_rate",
    ]]
    return batter_lookup, pair, cluster_audit, past


def fit_ridge():
    prediction = pd.read_parquet(
        MODEL_DIR / "insight_feature_ablation_predictions_success_screen_2024.parquet"
    )
    prediction = prediction.loc[prediction["model"].eq(MODEL_NAME)]
    features = pd.read_parquet(
        WORK / "oof" / "reverse_batter" / f"{OOF_CONFIG}.parquet"
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
    manual = (
        (imputer.transform(frame[FEATURES]) - scaler.mean_) / scaler.scale_
    ) @ ridge.coef_
    library = model.predict(frame[FEATURES])
    if not np.allclose(manual, library, atol=1e-8, rtol=1e-8):
        raise RuntimeError("Manual reverse-batter Ridge export mismatch")
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
    batter_lookup, pair, cluster_audit, past = build_final_tables(load_main())
    ridge = fit_ridge()
    batter_lookup.to_csv(FINAL_DIR / "reverse_batter_lookup_2025.csv", index=False)
    batter_lookup.to_parquet(FINAL_DIR / "reverse_batter_lookup_2025.parquet", index=False)
    pair.to_csv(FINAL_DIR / "reverse_batter_pair_table_2025.csv", index=False)
    pair.to_parquet(FINAL_DIR / "reverse_batter_pair_table_2025.parquet", index=False)
    (FINAL_DIR / "reverse_batter_ridge_correction.json").write_text(
        json.dumps(ridge, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "version": "reverse_batter_matchup_v1",
        "cutoff": CUTOFF,
        "main_trained_through": 2024,
        "center_mode": "season x pitcher_hand x batter_hand x count_state",
        "batter_algorithm": "kmeans",
        "batter_k_by_hand": {"left": 4, "right": 6},
        "smoothing": SMOOTHING,
        "half_life": HALF_LIFE,
        "pair_cells": int(len(pair)),
        "batter_lookup_rows": int(len(batter_lookup)),
        "valid_history_rows": int(len(past)),
        "cluster_audit": cluster_audit,
        "ridge": ridge,
    }
    (FINAL_DIR / "reverse_batter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
