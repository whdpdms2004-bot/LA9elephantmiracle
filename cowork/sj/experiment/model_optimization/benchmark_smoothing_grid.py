from __future__ import annotations

import gc
import json
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from benchmark_trackman500_fixed import enrich_trackman, feature_sets as tm_feature_sets
from benchmark_v2_ablation import TRIAL_NUMBER, encode_fold, load_frame, load_trial
from run_optuna_family import ROOT, SEED, TARGET, probability_metrics, recency_weights
from v2_temporal_features import ROW_RATE_SPECS


WORK_DIR = ROOT / "experiment" / "model_optimization"
STRENGTHS = [10, 50, 100, 200, 300, 500]


def add_missing_strengths(frame):
    output = frame.copy()
    for prefix, (rate_column, count_column, prior) in ROW_RATE_SPECS.items():
        rate = output[rate_column].astype(float)
        count = output[count_column].clip(lower=0).astype(float)
        for strength in [100, 300]:
            output[f"{prefix}_smoothed_{strength}"] = (
                (rate.fillna(prior) * count + prior * strength) / (count + strength)
            ).astype("float32")
            output[f"{prefix}_reliability_{strength}"] = (
                count / (count + strength)
            ).astype("float32")
    return output


def smoothing_sets(frame, original, v2_sets, tm_all):
    v1 = v2_sets["V1_BASE_RECHECK"]
    common = [
        column
        for column in frame
        if column.endswith("_is_missing")
        or "_n_eq_0" in column
        or "_n_le_" in column
        or column.startswith("pitcher_recent_")
        or column in {"pitcher_failure_rate_sum", "pitcher_control_component_gap"}
    ]
    output = {}
    for strength in STRENGTHS:
        selected = [
            column
            for column in frame
            if column.endswith(f"_smoothed_{strength}")
            or column.endswith(f"_reliability_{strength}")
        ]
        output[f"V2R{strength}_TM500_ALL"] = list(
            dict.fromkeys(original + v1 + common + selected + tm_all)
        )
    return output


def run_one(frame, features, version, fold, trial):
    started = time.time()
    params = dict(trial.params)
    half_life = float(params.pop("half_life"))
    train_mask, valid_mask, train_x, valid_x = encode_fold(frame, features, fold)
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    model = XGBClassifier(
        **params,
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
    row = {
        "experiment": f"xgboost_{version.lower()}",
        "family": "xgboost",
        "feature_version": version,
        "trial": TRIAL_NUMBER,
        "fold": fold,
        "train_through": fold - 1,
        "trackman": True,
        "trackman_cutoff": fold,
        "min_trackman_season_pitches": 500,
        "smoothing_strength": int(version.split("_", 1)[0].replace("V2R", "")),
        "feature_count": len(features),
        "best_iteration": int(model.best_iteration),
        "elapsed_sec": time.time() - started,
        **probability_metrics(valid_y, prediction),
    }
    pred = pd.DataFrame(
        {
            "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
            "season": fold,
            TARGET: valid_y,
            "model": row["experiment"],
            "prediction": prediction.astype("float32"),
        }
    )
    del model, train_x, valid_x, train_y, valid_y, weights, prediction
    gc.collect()
    return row, pred


def main():
    frame, original, v2_sets = load_frame()
    frame = add_missing_strengths(frame)
    tm = pd.read_parquet(WORK_DIR / "trackman500_asof_train.parquet")
    if not frame["row_id"].equals(tm["row_id"]):
        raise RuntimeError("Trackman cache row order mismatch")
    tm_columns = [column for column in tm if column not in {"row_id", "season"}]
    frame = pd.concat([frame, tm[tm_columns]], axis=1)
    before = set(frame.columns)
    frame = enrich_trackman(frame, tm_columns)
    enriched = tm_columns + [column for column in frame if column not in before]
    tm_all = tm_feature_sets(tm_columns, enriched)["TM500_ALL"]
    variants = smoothing_sets(frame, original, v2_sets, tm_all)
    trial = load_trial()
    results = []
    predictions = []
    screen = []
    for version, features in variants.items():
        row, pred = run_one(frame, features, version, 2024, trial)
        results.append(row)
        predictions.append(pred)
        screen.append((row["normalized_brier"], version))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    for _, version in sorted(screen)[:3]:
        row, pred = run_one(frame, variants[version], version, 2023, trial)
        results.append(row)
        predictions.append(pred)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    pd.DataFrame(results).to_csv(WORK_DIR / "smoothing_grid_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        WORK_DIR / "smoothing_grid_predictions.parquet", index=False
    )
    (WORK_DIR / "smoothing_grid_summary.json").write_text(
        json.dumps(
            {"strengths": STRENGTHS, "trial": TRIAL_NUMBER, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
