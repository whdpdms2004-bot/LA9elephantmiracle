from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
sys.path.insert(0, str(MODEL_DIR))

from benchmark_insight_features import (  # noqa: E402
    add_calibration_features,
    build_past_only_lookups,
    load_local_trial,
)
from benchmark_v2_ablation import encode_fold  # noqa: E402
from run_optuna_enhanced import load_enhanced_frame  # noqa: E402
from run_optuna_family import SEED, TARGET, probability_metrics, recency_weights  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    parser.add_argument("--folds", default="2024")
    parser.add_argument("--modes", default="all")
    return parser.parse_args()


def cluster_feature_columns(cache, mode):
    all_features = [column for column in cache if column not in {"row_id", "season"}]
    style = [column for column in all_features if column.startswith("pcm_style_")]
    hard = [
        column for column in all_features
        if column.startswith("pcm_h") or column == "pcm_cluster_id"
    ]
    meta = [
        column for column in all_features
        if column not in hard and column not in style
    ]
    if mode == "style":
        return style
    if mode == "soft_style":
        return list(dict.fromkeys(style + meta))
    if mode == "hard_style":
        return list(dict.fromkeys(style + hard + ["pcm_available"]))
    if mode == "all":
        return all_features
    raise ValueError(mode)


def run_model(frame, features, fold, trial, experiment):
    started = time.time()
    params = dict(trial.params)
    half_life = float(params.pop("half_life"))
    train_mask, valid_mask, train_x, valid_x = encode_fold(frame, features, fold)
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    model = XGBClassifier(
        **params,
        grow_policy="lossguide",
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=SEED + fold,
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
    metrics = probability_metrics(valid_y, prediction)
    row = {
        "experiment": experiment,
        "fold": fold,
        "feature_count": len(features),
        "best_iteration": int(model.best_iteration),
        "elapsed_sec": time.time() - started,
        **metrics,
    }
    pred = pd.DataFrame({
        "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
        "season": fold,
        TARGET: valid_y,
        "experiment": experiment,
        "prediction": prediction.astype("float32"),
    })
    del model, train_x, valid_x, train_y, valid_y, weights, prediction
    gc.collect()
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row, pred


def best_blend(pred, performance):
    joined = pred.merge(
        performance[["row_id", "prediction"]].rename(columns={"prediction": "performance"}),
        on="row_id", how="inner", validate="one_to_one",
    )
    if len(joined) != len(pred):
        return None
    best = None
    y = joined[TARGET].to_numpy("int8")
    candidate = joined["prediction"].to_numpy("float64")
    baseline = joined["performance"].to_numpy("float64")
    for weight in np.linspace(0.0, 1.0, 201):
        metrics = probability_metrics(y, weight * candidate + (1.0 - weight) * baseline)
        if best is None or metrics["brier"] < best["brier"]:
            best = {"cluster_weight": float(weight), **metrics}
    return best


def main():
    args = parse_args()
    configs = [value.strip() for value in args.configs.split(",") if value.strip()]
    folds = [int(value) for value in args.folds.split(",") if value]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    frame, base_features = load_enhanced_frame()
    labels = pd.read_parquet(MODEL_DIR / "failure_component_labels.parquet")
    lookups, _ = build_past_only_lookups(frame, labels)
    frame, _, prior_columns = add_calibration_features(frame, lookups)
    adjusted = [
        column for column in prior_columns
        if column.startswith(("pitcher_success_", "batter_success_"))
        and "_adjusted_smoothed_" in column
    ]
    if len(adjusted) != 2:
        raise RuntimeError(f"Expected two adjusted features, got {adjusted}")
    base_features = list(dict.fromkeys(base_features + adjusted))
    trial = load_local_trial()
    performance = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    performance = performance.loc[performance["track"].eq("performance")]
    results = []
    predictions = []
    for config in configs:
        cache = pd.read_parquet(WORK / "oof" / f"cluster_features_{config}.parquet")
        if not frame["row_id"].equals(cache["row_id"]):
            raise RuntimeError(f"Cluster cache misaligned: {config}")
        for mode in modes:
            cluster_columns = cluster_feature_columns(cache, mode)
            additions = cache[cluster_columns].copy()
            combined = pd.concat([frame, additions], axis=1)
            features = list(dict.fromkeys(base_features + cluster_columns))
            for fold in folds:
                experiment = f"cluster_{config}_{mode}"
                row, pred = run_model(combined, features, fold, trial, experiment)
                row.update({
                    "config_id": config,
                    "mode": mode,
                    "cluster_feature_count": len(cluster_columns),
                })
                blend = best_blend(pred, performance.loc[performance["season"].eq(fold)])
                if blend:
                    row.update({f"blend_{key}": value for key, value in blend.items()})
                results.append(row)
                predictions.append(pred)
            del combined, additions
            gc.collect()
        del cache
        gc.collect()
    result_path = WORK / "reports" / "cluster_feature_validation.csv"
    prediction_path = WORK / "oof" / "cluster_feature_predictions.parquet"
    pd.DataFrame(results).to_csv(result_path, index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(prediction_path, index=False)
    print(json.dumps({
        "results": str(result_path),
        "predictions": str(prediction_path),
        "runs": len(results),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
