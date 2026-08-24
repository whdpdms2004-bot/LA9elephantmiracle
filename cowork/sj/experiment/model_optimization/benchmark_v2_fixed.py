from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    ROOT,
    SEED,
    TARGET,
    probability_metrics,
    recency_weights,
)


WORK_DIR = ROOT / "experiment" / "model_optimization"
FOLDS = [2023, 2024]
SPECS = {
    "xgboost": ("xgboost_v1_full_2023_2024", 24),
    "catboost": ("catboost_v1_full_2023_2024", 71),
}


def load_trial(study_name, number):
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{(WORK_DIR / f'{study_name}.db').as_posix()}",
    )
    trial = next(item for item in study.trials if item.number == number)
    if trial.state != optuna.trial.TrialState.COMPLETE:
        raise RuntimeError(f"Incomplete trial: {study_name}/{number}")
    return trial


def load_v2_frame():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    cache = pd.read_parquet(WORK_DIR / "v2_temporal_train.parquet")
    if len(train) != len(cache) or cache["row_id"].duplicated().any():
        raise RuntimeError("Invalid V2 cache key")
    frame = train.merge(cache, on=["row_id", "season"], how="left", validate="one_to_one")
    added = [column for column in cache if column not in {"row_id", "season"}]
    if frame[added].isna().all().any():
        raise RuntimeError("Entirely missing V2 feature")
    features = [column for column in frame if column not in {"row_id", TARGET}]
    return frame, features, added


def encode_xgb(frame, features, fold):
    train_mask = frame["season"].lt(fold)
    valid_mask = frame["season"].eq(fold)
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
    return train_mask, valid_mask, train_x, valid_x


def main():
    frame, features, added = load_v2_frame()
    results = []
    prediction_parts = []
    print(f"frame={frame.shape} features={len(features)} added={len(added)}", flush=True)

    for family, (study_name, trial_number) in SPECS.items():
        trial = load_trial(study_name, trial_number)
        for fold_index, fold in enumerate(FOLDS):
            started = time.time()
            params = dict(trial.params)
            half_life = float(params.pop("half_life"))
            train_mask = frame["season"].lt(fold)
            valid_mask = frame["season"].eq(fold)
            weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
            target = frame.loc[valid_mask, TARGET].to_numpy("int8")

            if family == "xgboost":
                _, _, train_x, valid_x = encode_xgb(frame, features, fold)
                model = XGBClassifier(
                    **params,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    device="cuda",
                    random_state=SEED + fold_index,
                    n_jobs=6,
                    early_stopping_rounds=220,
                )
                model.fit(
                    train_x,
                    frame.loc[train_mask, TARGET].to_numpy("int8"),
                    sample_weight=weights,
                    eval_set=[(valid_x, target)],
                    verbose=False,
                )
                prediction = model.predict_proba(valid_x)[:, 1]
                best_iteration = int(model.best_iteration)
                del train_x, valid_x
            else:
                cat_frame = frame[features].copy()
                for column in CATEGORICAL_COLUMNS:
                    cat_frame[column] = cat_frame[column].fillna("__MISSING__").astype(str)
                train_pool = Pool(
                    cat_frame.loc[train_mask],
                    label=frame.loc[train_mask, TARGET],
                    cat_features=CATEGORICAL_COLUMNS,
                    weight=weights,
                )
                valid_pool = Pool(
                    cat_frame.loc[valid_mask],
                    label=target,
                    cat_features=CATEGORICAL_COLUMNS,
                )
                model = CatBoostClassifier(
                    **params,
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    task_type="GPU",
                    devices="0",
                    random_seed=SEED + fold_index,
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
                best_iteration = int(model.get_best_iteration())
                del cat_frame, train_pool, valid_pool

            metrics = probability_metrics(target, prediction)
            row = {
                "experiment": f"{family}_v2_fixed_trial_{trial_number}",
                "family": family,
                "feature_version": "V2_TEMPORAL",
                "trial": trial_number,
                "fold": fold,
                "train_through": fold - 1,
                "trackman": False,
                "best_iteration": best_iteration,
                "elapsed_sec": time.time() - started,
                **metrics,
            }
            results.append(row)
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
                        "season": fold,
                        TARGET: target,
                        "model": row["experiment"],
                        "prediction": prediction.astype("float32"),
                    }
                )
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)
            del model, prediction, target, weights
            gc.collect()

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(WORK_DIR / "v2_fixed_results.csv", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_parquet(
        WORK_DIR / "v2_fixed_predictions.parquet", index=False
    )
    summary = {
        "feature_count": len(features),
        "added_feature_count": len(added),
        "results": results,
    }
    (WORK_DIR / "v2_fixed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
