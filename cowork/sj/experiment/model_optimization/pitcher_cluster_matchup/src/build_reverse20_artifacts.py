"""Build 2025 inference artifacts for the validated 20-seed reverse bag."""

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
MODEL_OPT = ROOT / "experiment" / "model_optimization"
WORK = MODEL_OPT / "pitcher_cluster_matchup"
FINAL_DIR = WORK / "final" / "reverse_seedbag20"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(MODEL_OPT))

from run_reverse_seedbag20 import (  # noqa: E402
    BATTER_K,
    CACHE,
    FEATURES,
    HALF_LIFE,
    RIDGE_ALPHA,
    SEEDS,
    SMOOTHING,
)
from screen_reverse_batter_clusters import (  # noqa: E402
    add_context_residual,
    add_pitcher_type,
    build_batter_profile,
    cluster_batters,
    load_main,
)


CUTOFF = 2025
MODEL_NAME = "xgboost_insight_insight_success_adjusted"


def pair_table(past: pd.DataFrame, batter_lookup: pd.DataFrame) -> pd.DataFrame:
    typed = past.merge(
        batter_lookup[["batter_id", "batter_hand", "batter_type"]],
        on=["batter_id", "batter_hand"], how="left", validate="many_to_one",
    )
    typed["batter_type"] = typed["batter_type"].fillna(
        "RBH" + typed["batter_hand"].astype(str) + "_new"
    )
    weight = np.power(0.5, (CUTOFF - typed["season"].to_numpy(float)) / HALF_LIFE)
    work = typed[["pitcher_type", "batter_type"]].copy()
    work["weighted_residual"] = typed["reverse_residual"].to_numpy(float) * weight
    work["weighted_reverse"] = typed["reverse"].to_numpy(float) * weight
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
    return pair[[
        "pitcher_type", "batter_type", "effective_n", "raw_n",
        "reverse_pair_delta", "reverse_pair_delta_reliability", "reverse_pair_rate",
    ]]


def ridge_artifact(seed: int) -> dict:
    prediction = pd.read_parquet(
        MODEL_OPT / "insight_feature_ablation_predictions_success_screen_2024.parquet"
    )
    prediction = prediction.loc[prediction["model"].eq(MODEL_NAME)]
    features = pd.read_parquet(CACHE / f"seed_{seed}.parquet")
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
        raise RuntimeError(f"Manual Ridge export mismatch for seed {seed}")
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


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    typed = add_pitcher_type(load_main(), CUTOFF)
    past = add_context_residual(typed)
    profile = build_batter_profile(past)
    records = []
    for index, seed in enumerate(SEEDS):
        prefix = f"reverse20_s{seed}"
        lookup_file = FINAL_DIR / f"{prefix}_lookup.csv"
        pair_file = FINAL_DIR / f"{prefix}_pair.csv"
        ridge_file = FINAL_DIR / f"{prefix}_ridge.json"
        if lookup_file.exists() and pair_file.exists() and ridge_file.exists():
            lookup = pd.read_csv(lookup_file)
            pair = pd.read_csv(pair_file)
            cluster_audit = {"cached": True}
            ridge = json.loads(ridge_file.read_text(encoding="utf-8"))
        else:
            lookup, cluster_audit = cluster_batters(
                profile, "kmeans", BATTER_K, seed=seed
            )
            pair = pair_table(past, lookup)
            ridge = ridge_artifact(seed)
            lookup.to_csv(lookup_file, index=False)
            pair.to_csv(pair_file, index=False)
            ridge_file.write_text(
                json.dumps(ridge, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        records.append(
            {
                "seed": seed,
                "prefix": prefix,
                "lookup_rows": int(len(lookup)),
                "pair_cells": int(len(pair)),
                "cluster_audit": cluster_audit,
                "ridge": ridge,
            }
        )
        print(f"final reverse artifact {index + 1:02d}/20 seed={seed}", flush=True)
    manifest = {
        "version": "reverse_batter_seedbag20_v1",
        "cutoff": CUTOFF,
        "trained_through": 2024,
        "seeds": SEEDS,
        "batter_k_by_hand": {"left": BATTER_K[0], "right": BATTER_K[1]},
        "smoothing": SMOOTHING,
        "half_life": HALF_LIFE,
        "ridge_alpha": RIDGE_ALPHA,
        "records": records,
    }
    (FINAL_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"seeds": SEEDS, "artifact_files": 60}, ensure_ascii=False))


if __name__ == "__main__":
    main()
