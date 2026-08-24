from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from benchmark_v2_ablation import encode_fold
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_lightgbm_enhanced import brier_eval
from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    ROOT,
    SEED,
    TARGET,
    probability_metrics,
    recency_weights,
)


WORK_DIR = ROOT / "experiment" / "model_optimization"
PART_DIR = WORK_DIR / "enhanced_seed_oof_parts"
FOLDS = [2022, 2023, 2024]
SEED_OFFSETS = [0, 100_000, 200_000]
STUDY_SPECS = [
    {
        "family": "xgboost",
        "study": "xgboost_v2r200_tm500_robust",
        "slug": "xgb_robust",
        "top_objective": 3,
        "top_recent": 3,
    },
    {
        "family": "xgboost",
        "study": "xgboost_v2r200_tm500_2024",
        "slug": "xgb_recent",
        "top_objective": 4,
        "top_recent": 0,
    },
    {
        "family": "xgboost",
        "study": "xgboost_v2r200_tm500_local_2024",
        "slug": "xgb_local",
        "top_objective": 4,
        "top_recent": 0,
    },
    {
        "family": "lightgbm",
        "study": "lightgbm_v2r200_tm500_robust",
        "slug": "lgb_robust",
        "top_objective": 3,
        "top_recent": 3,
    },
    {
        "family": "catboost",
        "study": "catboost_v2r200_tm500_robust",
        "slug": "cat_robust",
        "top_objective": 3,
        "top_recent": 3,
    },
]


def load_study(spec):
    database = WORK_DIR / f"{spec['study']}.db"
    if not database.is_file():
        print(f"skip missing study: {spec['study']}", flush=True)
        return None
    return optuna.load_study(
        study_name=spec["study"], storage=f"sqlite:///{database.as_posix()}"
    )


def select_trials(spec):
    study = load_study(spec)
    if study is None:
        return []
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        return []
    objective = sorted(complete, key=lambda trial: trial.value)[
        : spec["top_objective"]
    ]
    recent = sorted(
        complete,
        key=lambda trial: trial.user_attrs.get("fold_2024", {}).get(
            "normalized_brier", np.inf
        ),
    )[: spec["top_recent"]]
    output = []
    seen = set()
    for trial in objective + recent:
        if trial.number in seen:
            continue
        seen.add(trial.number)
        output.append(
            {
                "family": spec["family"],
                "study": spec["study"],
                "trial": trial.number,
                "objective": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
                "model_name": f"{spec['slug']}_t{trial.number}_seedbag3",
            }
        )
    return output


def safe_stem(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def part_paths(model_name, fold):
    stem = safe_stem(f"{model_name}_fold{fold}")
    return PART_DIR / f"{stem}.parquet", PART_DIR / f"{stem}.json"


def run_xgboost_candidate(frame, features, candidate, fold):
    started = time.time()
    train_mask, valid_mask, train_x, valid_x = encode_fold(frame, features, fold)
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    params = dict(candidate["params"])
    half_life = float(params.pop("half_life"))
    weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    seed_predictions = []
    iterations = []
    for seed_offset in SEED_OFFSETS:
        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda",
            random_state=SEED + candidate["trial"] + fold + seed_offset,
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
        seed_predictions.append(model.predict_proba(valid_x)[:, 1])
        iterations.append(int(model.best_iteration))
        del model
        gc.collect()
    prediction = np.mean(seed_predictions, axis=0).astype("float32")
    metrics = probability_metrics(valid_y, prediction)
    metadata = {
        **{key: value for key, value in candidate.items() if key != "user_attrs"},
        "fold": fold,
        "train_through": fold - 1,
        "feature_version": "V2R200_TM500_ALL",
        "feature_count": len(features),
        "seeds": [SEED + candidate["trial"] + fold + x for x in SEED_OFFSETS],
        "best_iterations": iterations,
        "elapsed_sec": time.time() - started,
        **metrics,
    }
    prediction_frame = pd.DataFrame(
        {
            "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
            "season": fold,
            TARGET: valid_y,
            "model": candidate["model_name"],
            "prediction": prediction,
        }
    )
    del train_x, valid_x, train_y, valid_y, weights, seed_predictions, prediction
    gc.collect()
    return prediction_frame, metadata


def run_lightgbm_candidate(frame, features, candidate, fold):
    started = time.time()
    train_mask, valid_mask, train_x, valid_x = encode_fold(frame, features, fold)
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    params = dict(candidate["params"])
    half_life = float(params.pop("half_life"))
    weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    seed_predictions = []
    iterations = []
    for seed_offset in SEED_OFFSETS:
        model = LGBMClassifier(
            **params,
            objective="binary",
            metric="None",
            random_state=SEED + candidate["trial"] + fold + seed_offset,
            n_jobs=6,
            verbosity=-1,
            force_col_wise=True,
        )
        model.fit(
            train_x,
            train_y,
            sample_weight=weights,
            eval_set=[(valid_x, valid_y)],
            eval_metric=brier_eval,
            callbacks=[lgb.early_stopping(220, first_metric_only=True, verbose=False)],
        )
        seed_predictions.append(
            model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]
        )
        iterations.append(int(model.best_iteration_))
        del model
        gc.collect()
    prediction = np.mean(seed_predictions, axis=0).astype("float32")
    metrics = probability_metrics(valid_y, prediction)
    metadata = {
        **{key: value for key, value in candidate.items() if key != "user_attrs"},
        "fold": fold,
        "train_through": fold - 1,
        "feature_version": "V2R200_TM500_ALL",
        "feature_count": len(features),
        "seeds": [SEED + candidate["trial"] + fold + x for x in SEED_OFFSETS],
        "best_iterations": iterations,
        "elapsed_sec": time.time() - started,
        **metrics,
    }
    prediction_frame = pd.DataFrame(
        {
            "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
            "season": fold,
            TARGET: valid_y,
            "model": candidate["model_name"],
            "prediction": prediction,
        }
    )
    del train_x, valid_x, train_y, valid_y, weights, seed_predictions, prediction
    gc.collect()
    return prediction_frame, metadata


def prepare_cat_frame(frame, features):
    output = frame[features].copy()
    for column in CATEGORICAL_COLUMNS:
        output[column] = output[column].fillna("__MISSING__").astype(str)
    return output


def run_catboost_candidate(frame, cat_frame, candidate, fold):
    started = time.time()
    train_mask = frame["season"].lt(fold)
    valid_mask = frame["season"].eq(fold)
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    params = dict(candidate["params"])
    half_life = float(params.pop("half_life"))
    weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    seed_predictions = []
    iterations = []
    for seed_offset in SEED_OFFSETS:
        train_pool = Pool(
            cat_frame.loc[train_mask],
            label=train_y,
            cat_features=CATEGORICAL_COLUMNS,
            weight=weights,
        )
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
            random_seed=SEED + candidate["trial"] + fold + seed_offset,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=220,
        )
        seed_predictions.append(model.predict_proba(valid_pool)[:, 1])
        iterations.append(int(model.get_best_iteration()))
        del model, train_pool, valid_pool
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    prediction = np.mean(seed_predictions, axis=0).astype("float32")
    metrics = probability_metrics(valid_y, prediction)
    metadata = {
        **{key: value for key, value in candidate.items() if key != "user_attrs"},
        "fold": fold,
        "train_through": fold - 1,
        "feature_version": "V2R200_TM500_ALL",
        "feature_count": cat_frame.shape[1],
        "seeds": [SEED + candidate["trial"] + fold + x for x in SEED_OFFSETS],
        "best_iterations": iterations,
        "elapsed_sec": time.time() - started,
        **metrics,
    }
    prediction_frame = pd.DataFrame(
        {
            "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
            "season": fold,
            TARGET: valid_y,
            "model": candidate["model_name"],
            "prediction": prediction,
        }
    )
    del train_y, valid_y, weights, seed_predictions, prediction
    gc.collect()
    return prediction_frame, metadata


def main():
    PART_DIR.mkdir(parents=True, exist_ok=True)
    candidates = []
    for spec in STUDY_SPECS:
        candidates.extend(select_trials(spec))
    if not candidates:
        raise RuntimeError("No completed enhanced trials were found")
    (WORK_DIR / "enhanced_seed_oof_selection.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    frame, features = load_enhanced_frame()
    cat_frame = None

    for candidate in candidates:
        if candidate["family"] == "catboost" and cat_frame is None:
            cat_frame = prepare_cat_frame(frame, features)
        for fold in FOLDS:
            parquet_path, metadata_path = part_paths(candidate["model_name"], fold)
            if parquet_path.is_file() and metadata_path.is_file():
                print(f"resume skip {parquet_path.name}", flush=True)
                continue
            print(
                json.dumps(
                    {
                        "event": "start",
                        "model": candidate["model_name"],
                        "fold": fold,
                    }
                ),
                flush=True,
            )
            if candidate["family"] == "xgboost":
                prediction, metadata = run_xgboost_candidate(
                    frame, features, candidate, fold
                )
            elif candidate["family"] == "lightgbm":
                prediction, metadata = run_lightgbm_candidate(
                    frame, features, candidate, fold
                )
            else:
                prediction, metadata = run_catboost_candidate(
                    frame, cat_frame, candidate, fold
                )
            prediction.to_parquet(parquet_path, index=False)
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(metadata, ensure_ascii=False), flush=True)

    metadata_rows = []
    prediction_parts = []
    for metadata_path in sorted(PART_DIR.glob("*.json")):
        metadata_rows.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    for parquet_path in sorted(PART_DIR.glob("*.parquet")):
        prediction_parts.append(pd.read_parquet(parquet_path))
    pd.DataFrame(metadata_rows).to_csv(
        WORK_DIR / "enhanced_seed_oof_metrics.csv", index=False
    )
    pd.concat(prediction_parts, ignore_index=True).to_parquet(
        WORK_DIR / "enhanced_seed_oof_predictions.parquet", index=False
    )
    manifest = {
        "feature_version": "V2R200_TM500_ALL",
        "folds": FOLDS,
        "seed_offsets": SEED_OFFSETS,
        "candidates": candidates,
        "part_directory": str(PART_DIR.relative_to(ROOT)),
        "metrics": "experiment/model_optimization/enhanced_seed_oof_metrics.csv",
        "predictions": "experiment/model_optimization/enhanced_seed_oof_predictions.parquet",
        "rules": {
            "validation": "train season < fold, validate season == fold",
            "trackman": "strict as-of cutoff; pitcher-season >=500 only",
        },
    }
    (WORK_DIR / "enhanced_seed_oof_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
