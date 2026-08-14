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
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))

from build_matchup_features import (  # noqa: E402
    build_batter_profile,
    cluster_batters,
    pitcher_lookup,
    shrink_table,
)


CUTOFF = 2025
PITCHER_CONFIG = "com_gmm_p8_l2r4_7de7125e"
MATCHUP_CONFIG = "m_com_gmm_kme_b3r4_l1000_h1_d72788b5"
BATTER_ALGORITHM = "kmeans"
BATTER_K = (3, 4)
PROFILE_LAMBDA = 200.0
SMOOTHING = 1000.0
HALF_LIFE = 1.0
RIDGE_ALPHA = 10.0
FEATURES = [
    "match_pair_delta", "match_pair_delta_reliability",
    "match_pair_delta_rate", "match_pair_known",
]


def build_final_tables(main):
    p_lookup = pitcher_lookup(PITCHER_CONFIG, CUTOFF)
    past = main.merge(
        p_lookup[["pitcher_id", "pitcher_type"]],
        on="pitcher_id", how="left", validate="many_to_one",
    )
    past["pitcher_type"] = past["pitcher_type"].fillna(
        "H" + past["pitcher_hand"].astype(str) + "_new"
    )
    batter_profile, past = build_batter_profile(past, PROFILE_LAMBDA)
    b_lookup, batter_audit = cluster_batters(
        batter_profile, BATTER_ALGORITHM, BATTER_K, pca_dim=8
    )
    past = past.merge(
        b_lookup[["batter_id", "batter_hand", "batter_type"]],
        on=["batter_id", "batter_hand"], how="left", validate="many_to_one",
    )
    past["batter_type"] = past["batter_type"].fillna(
        "BH" + past["batter_hand"].astype(str) + "_new"
    )
    past["recency_weight"] = np.power(
        0.5, (CUTOFF - past["season"].to_numpy("float64")) / HALF_LIFE
    )
    past["weighted_residual"] = past["success_residual"] * past["recency_weight"]
    past["weighted_success"] = past["control_success"] * past["recency_weight"]
    pair = shrink_table(
        past, ["pitcher_type", "batter_type"], "match_pair_delta",
        SMOOTHING, "recency_weight",
    )
    pitcher_bhand = shrink_table(
        past, ["pitcher_type", "batter_hand"], "match_pitcher_bhand_delta",
        SMOOTHING, "recency_weight",
    )
    phand_batter = shrink_table(
        past, ["pitcher_hand", "batter_type"], "match_phand_batter_delta",
        SMOOTHING, "recency_weight",
    )
    return p_lookup, b_lookup, pair, pitcher_bhand, phand_batter, batter_audit


def fit_ridge_artifact():
    prediction = pd.read_parquet(
        MODEL_DIR / "insight_feature_ablation_predictions_success_screen_2024.parquet"
    )
    prediction = prediction.loc[
        prediction["model"].eq("xgboost_insight_insight_success_adjusted")
    ]
    cache = pd.read_parquet(WORK / "oof" / f"matchup_features_{MATCHUP_CONFIG}.parquet")
    frame = prediction.merge(cache, on=["row_id", "season"], validate="one_to_one")
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
        raise RuntimeError("Manual Ridge export does not reproduce sklearn")
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


def sample_feature_join(test, p_lookup, b_lookup, pair):
    sample = test[["row_id", "pitcher_id", "pitcher_hand", "batter_id", "batter_hand"]].copy()
    sample = sample.merge(
        p_lookup[["pitcher_id", "pitcher_type"]], on="pitcher_id", how="left"
    )
    sample["pitcher_type"] = sample["pitcher_type"].fillna(
        "H" + sample["pitcher_hand"].astype(str) + "_new"
    )
    sample = sample.merge(
        b_lookup[["batter_id", "batter_hand", "batter_type"]],
        on=["batter_id", "batter_hand"], how="left",
    )
    sample["batter_type"] = sample["batter_type"].fillna(
        "BH" + sample["batter_hand"].astype(str) + "_new"
    )
    sample = sample.merge(pair, on=["pitcher_type", "batter_type"], how="left")
    sample["match_pair_known"] = sample["match_pair_delta"].notna().astype("float32")
    sample["match_pair_delta"] = sample["match_pair_delta"].fillna(0.0)
    sample["match_pair_delta_reliability"] = sample[
        "match_pair_delta_reliability"
    ].fillna(0.0)
    return sample[["row_id", *FEATURES]]


def main():
    output = WORK / "final" / "robust_matchup_v1"
    output.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(ROOT / "data" / "train.csv", usecols=[
        "row_id", "season", "pitcher_id", "pitcher_hand", "batter_id",
        "batter_hand", "control_success",
    ])
    p_lookup, b_lookup, pair, pitcher_bhand, phand_batter, batter_audit = build_final_tables(main)
    p_lookup[["pitcher_id", "pitcher_hand", "pitcher_type", "cohort"]].to_parquet(
        output / "pitcher_lookup_2025.parquet", index=False
    )
    p_lookup[["pitcher_id", "pitcher_hand", "pitcher_type", "cohort"]].to_csv(
        output / "pitcher_lookup_2025.csv", index=False
    )
    b_lookup.to_parquet(output / "batter_lookup_2025.parquet", index=False)
    b_lookup.to_csv(output / "batter_lookup_2025.csv", index=False)
    pair.to_parquet(output / "pair_table_2025.parquet", index=False)
    pair.to_csv(output / "pair_table_2025.csv", index=False)
    pitcher_bhand.to_parquet(output / "pitcher_bhand_table_2025.parquet", index=False)
    phand_batter.to_parquet(output / "phand_batter_table_2025.parquet", index=False)
    ridge = fit_ridge_artifact()
    (output / "ridge_correction.json").write_text(
        json.dumps(ridge, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sample = pd.read_csv(ROOT / "data" / "test.csv")
    smoke = sample_feature_join(sample, p_lookup, b_lookup, pair)
    smoke.to_csv(output / "sample_matchup_features.csv", index=False)
    manifest = {
        "version": "robust_matchup_v1",
        "cutoff": CUTOFF,
        "main_trained_through": 2024,
        "trackman_trained_through": 2024,
        "min_trackman_season_pitches": 500,
        "pitcher_config": PITCHER_CONFIG,
        "pitcher_hand_clusters": {"Left": 2, "Right": 4},
        "batter_algorithm": BATTER_ALGORITHM,
        "batter_hand_clusters": {"Left": 3, "Right": 4},
        "smoothing": SMOOTHING,
        "half_life": HALF_LIFE,
        "pair_cells": int(len(pair)),
        "pitcher_lookup_rows": int(len(p_lookup)),
        "batter_lookup_rows": int(len(b_lookup)),
        "batter_cluster_audit": batter_audit,
        "ridge": ridge,
        "inference_rule": "Frozen lookup only; no test-batch aggregation or refitting.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "pair_cells": len(pair),
        "pitchers": len(p_lookup),
        "batters": len(b_lookup),
        "sample_rows": len(smoke),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
