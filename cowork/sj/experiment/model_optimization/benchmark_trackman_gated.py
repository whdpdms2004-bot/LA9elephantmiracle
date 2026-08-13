from __future__ import annotations

import gc
import json
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from benchmark_v2_ablation import TRIAL_NUMBER, load_frame, load_trial
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    ROOT,
    SEED,
    TARGET,
    probability_metrics,
    recency_weights,
)


WORK_DIR = ROOT / "experiment" / "model_optimization"


def encode_subset(frame, features, train_mask, valid_mask):
    train_x = frame.loc[train_mask, features].copy()
    valid_x = frame.loc[valid_mask, features].copy()
    for column in CATEGORICAL_COLUMNS:
        values = train_x[column].fillna("__MISSING__").astype(str)
        mapping = {value: index for index, value in enumerate(pd.unique(values))}
        train_x[column] = values.map(mapping).astype("int32")
        valid_x[column] = (
            valid_x[column]
            .fillna("__MISSING__")
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype("int32")
        )
    for column in features:
        train_x[column] = pd.to_numeric(train_x[column], errors="coerce").astype("float32")
        valid_x[column] = pd.to_numeric(valid_x[column], errors="coerce").astype("float32")
    return train_x, valid_x


def train_subset(frame, features, train_mask, valid_mask, fold, trial, seed_offset):
    params = dict(trial.params)
    half_life = float(params.pop("half_life"))
    train_x, valid_x = encode_subset(frame, features, train_mask, valid_mask)
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=SEED + fold + seed_offset,
        n_jobs=6,
        early_stopping_rounds=220,
    )
    model.fit(
        train_x,
        train_y,
        sample_weight=weights,
        eval_set=[(valid_x, valid_y)],
        verbose=False,
    )
    prediction = model.predict_proba(valid_x)[:, 1]
    iteration = int(model.best_iteration)
    del model, train_x, valid_x, train_y, valid_y, weights
    gc.collect()
    return prediction, iteration


def run_gated(frame, available_features, unavailable_features, version, fold, trial):
    started = time.time()
    train_base = frame["season"].lt(fold)
    valid_base = frame["season"].eq(fold)
    available = frame["tm500_available"].eq(1)
    train_available = train_base & available
    valid_available = valid_base & available
    train_unavailable = train_base & ~available
    valid_unavailable = valid_base & ~available

    available_prediction, available_iteration = train_subset(
        frame,
        available_features,
        train_available,
        valid_available,
        fold,
        trial,
        101,
    )
    unavailable_prediction, unavailable_iteration = train_subset(
        frame,
        unavailable_features,
        train_unavailable,
        valid_unavailable,
        fold,
        trial,
        202,
    )
    valid_index = frame.index[valid_base]
    prediction = pd.Series(index=valid_index, dtype="float64")
    prediction.loc[frame.index[valid_available]] = available_prediction
    prediction.loc[frame.index[valid_unavailable]] = unavailable_prediction
    if prediction.isna().any():
        raise RuntimeError("Gated model left validation rows unpredicted")
    valid_y = frame.loc[valid_base, TARGET].to_numpy("int8")
    overall = probability_metrics(valid_y, prediction.to_numpy())
    available_metrics = probability_metrics(
        frame.loc[valid_available, TARGET], available_prediction
    )
    unavailable_metrics = probability_metrics(
        frame.loc[valid_unavailable, TARGET], unavailable_prediction
    )
    experiment = f"xgboost_{version.lower()}"
    row = {
        "experiment": experiment,
        "family": "xgboost_gated",
        "feature_version": version,
        "trial": TRIAL_NUMBER,
        "fold": fold,
        "train_through": fold - 1,
        "trackman": True,
        "trackman_cutoff": fold,
        "min_trackman_season_pitches": 500,
        "available_feature_count": len(available_features),
        "unavailable_feature_count": len(unavailable_features),
        "available_train_rows": int(train_available.sum()),
        "unavailable_train_rows": int(train_unavailable.sum()),
        "available_valid_rows": int(valid_available.sum()),
        "unavailable_valid_rows": int(valid_unavailable.sum()),
        "available_best_iteration": available_iteration,
        "unavailable_best_iteration": unavailable_iteration,
        "available_brier": available_metrics["brier"],
        "available_bss": available_metrics["bss"],
        "available_auc": available_metrics["auc"],
        "unavailable_brier": unavailable_metrics["brier"],
        "unavailable_bss": unavailable_metrics["bss"],
        "unavailable_auc": unavailable_metrics["auc"],
        "elapsed_sec": time.time() - started,
        **overall,
    }
    pred = pd.DataFrame(
        {
            "row_id": frame.loc[valid_base, "row_id"].to_numpy(),
            "season": fold,
            TARGET: valid_y,
            "model": experiment,
            "tm500_available": frame.loc[valid_base, "tm500_available"].to_numpy("int8"),
            "prediction": prediction.to_numpy("float32"),
        }
    )
    return row, pred


def main():
    frame, enhanced_features = load_enhanced_frame()
    _, original, v2_sets = load_frame()
    v2_features = list(dict.fromkeys(original + v2_sets["V2_ROW_SELECTED_200"]))
    trial = load_trial()
    variants = {
        "GATED_TM_AVAILABLE_V2_UNAVAILABLE": (enhanced_features, v2_features),
        "GATED_SEPARATE_ENHANCED": (enhanced_features, enhanced_features),
    }
    results = []
    predictions = []
    for fold in [2023, 2024]:
        for version, (available_features, unavailable_features) in variants.items():
            row, pred = run_gated(
                frame,
                available_features,
                unavailable_features,
                version,
                fold,
                trial,
            )
            results.append(row)
            predictions.append(pred)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    pd.DataFrame(results).to_csv(WORK_DIR / "trackman_gated_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        WORK_DIR / "trackman_gated_predictions.parquet", index=False
    )
    (WORK_DIR / "trackman_gated_summary.json").write_text(
        json.dumps(
            {"trial": TRIAL_NUMBER, "variants": list(variants), "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
