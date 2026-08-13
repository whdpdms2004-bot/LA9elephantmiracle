from __future__ import annotations

import gc
import json
import time

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool

from benchmark_trackman500_fixed import enrich_trackman, feature_sets
from benchmark_v2_ablation import load_frame
from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    ROOT,
    SEED,
    TARGET,
    probability_metrics,
    recency_weights,
)


WORK_DIR = ROOT / "experiment" / "model_optimization"
STUDY_NAME = "catboost_v1_full_2023_2024"
TRIAL_NUMBER = 71


def load_trial():
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{(WORK_DIR / f'{STUDY_NAME}.db').as_posix()}",
    )
    return next(item for item in study.trials if item.number == TRIAL_NUMBER)


def run_one(frame, original, additions, feature_version, fold, trial):
    started = time.time()
    features = list(dict.fromkeys(original + additions))
    train_mask = frame["season"].lt(fold)
    valid_mask = frame["season"].eq(fold)
    cat_frame = frame[features].copy()
    for column in CATEGORICAL_COLUMNS:
        cat_frame[column] = cat_frame[column].fillna("__MISSING__").astype(str)
    params = dict(trial.params)
    half_life = float(params.pop("half_life"))
    weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    train_pool = Pool(
        cat_frame.loc[train_mask],
        label=frame.loc[train_mask, TARGET],
        cat_features=CATEGORICAL_COLUMNS,
        weight=weights,
    )
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    valid_pool = Pool(
        cat_frame.loc[valid_mask],
        label=valid_y,
        cat_features=CATEGORICAL_COLUMNS,
    )
    model = CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type="GPU",
        devices="0",
        random_seed=SEED + fold,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=220,
    )
    prediction = model.predict_proba(valid_pool)[:, 1]
    experiment = f"catboost_{feature_version.lower()}"
    row = {
        "experiment": experiment,
        "family": "catboost",
        "feature_version": feature_version,
        "trial": TRIAL_NUMBER,
        "fold": fold,
        "train_through": fold - 1,
        "trackman": "TM500" in feature_version,
        "trackman_cutoff": fold if "TM500" in feature_version else None,
        "min_trackman_season_pitches": 500 if "TM500" in feature_version else None,
        "feature_count": len(features),
        "best_iteration": int(model.get_best_iteration()),
        "elapsed_sec": time.time() - started,
        **probability_metrics(valid_y, prediction),
    }
    pred = pd.DataFrame(
        {
            "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
            "season": fold,
            TARGET: valid_y,
            "model": experiment,
            "prediction": prediction.astype("float32"),
        }
    )
    del model, train_pool, valid_pool, cat_frame, weights, prediction, valid_y
    gc.collect()
    return row, pred


def main():
    frame, original, v2_sets = load_frame()
    tm = pd.read_parquet(WORK_DIR / "trackman500_asof_train.parquet")
    if not frame["row_id"].equals(tm["row_id"]):
        raise RuntimeError("Trackman cache row order mismatch")
    tm_columns = [column for column in tm if column not in {"row_id", "season"}]
    frame = pd.concat([frame, tm[tm_columns]], axis=1)
    before = set(frame.columns)
    frame = enrich_trackman(frame, tm_columns)
    enriched_columns = tm_columns + [column for column in frame if column not in before]
    tm_all = feature_sets(tm_columns, enriched_columns)["TM500_ALL"]
    variants = {
        "V1_RECHECK": v2_sets["V1_BASE_RECHECK"],
        "V2R200": v2_sets["V2_ROW_SELECTED_200"],
        "V1_TM500_ALL": v2_sets["V1_BASE_RECHECK"] + tm_all,
        "V2R200_TM500_ALL": v2_sets["V2_ROW_SELECTED_200"] + tm_all,
    }
    trial = load_trial()
    results = []
    predictions = []
    screen = []
    for name, additions in variants.items():
        row, pred = run_one(frame, original, additions, name, 2024, trial)
        results.append(row)
        predictions.append(pred)
        screen.append((row["normalized_brier"], name))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    for _, name in sorted(screen)[:2]:
        row, pred = run_one(frame, original, variants[name], name, 2023, trial)
        results.append(row)
        predictions.append(pred)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    pd.DataFrame(results).to_csv(WORK_DIR / "cat_enhanced_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        WORK_DIR / "cat_enhanced_predictions.parquet", index=False
    )
    (WORK_DIR / "cat_enhanced_summary.json").write_text(
        json.dumps({"trial": TRIAL_NUMBER, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
