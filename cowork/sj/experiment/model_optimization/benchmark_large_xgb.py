from __future__ import annotations

import argparse
import gc
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from xgboost import XGBClassifier

from benchmark_insight_features import (
    WORK_DIR,
    add_calibration_features,
    build_past_only_lookups,
)
from benchmark_v2_ablation import encode_fold
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import SEED, TARGET, probability_metrics, recency_weights


STUDY_NAME = "xgboost_v2r200_tm500_local_2024"
TRIAL_NUMBER = 93
FEATURE_VERSION = "INSIGHT_SUCCESS_ADJUSTED"
OUTPUT_DIR = WORK_DIR / "large_xgb"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
RESULT_PATH = OUTPUT_DIR / "large_xgb_results.csv"
MANIFEST_PATH = OUTPUT_DIR / "large_xgb_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="2024")
    parser.add_argument("--candidates", default="all")
    parser.add_argument("--seed-offsets", default="0")
    parser.add_argument("--early-stopping-rounds", type=int, default=600)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_trial93():
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{(WORK_DIR / f'{STUDY_NAME}.db').as_posix()}",
    )
    return next(item for item in study.trials if item.number == TRIAL_NUMBER)


def load_frame_and_features():
    frame, base_features = load_enhanced_frame()
    labels = pd.read_parquet(WORK_DIR / "failure_component_labels.parquet")
    lookups, audit = build_past_only_lookups(frame, labels)
    if not all(
        item["source_season"] is None
        or item["source_season"] < item["target_season"]
        for item in audit
    ):
        raise RuntimeError("Past-only feature audit failed")
    frame, _, prior_columns = add_calibration_features(frame, lookups)
    adjusted = [
        column
        for column in prior_columns
        if column.startswith(("pitcher_success_", "batter_success_"))
        and "_adjusted_smoothed_" in column
    ]
    if len(adjusted) != 2:
        raise RuntimeError(f"Expected two adjusted success features, got {adjusted}")
    features = list(dict.fromkeys(base_features + adjusted))
    return frame, features


def candidate_grid(base_params: dict):
    base = deepcopy(base_params)
    base.pop("half_life", None)

    def make(
        name: str,
        *,
        learning_rate: float | None = None,
        n_estimators: int | None = None,
        max_depth: int | None = None,
        max_leaves: int | None = None,
        min_child_weight: float | None = None,
        reg_lambda: float | None = None,
        reg_alpha: float | None = None,
        max_bin: int | None = None,
        subsample: float | None = None,
        colsample_bytree: float | None = None,
        eval_metric: str = "rmse",
    ):
        params = deepcopy(base)
        overrides = {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_leaves": max_leaves,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "max_bin": max_bin,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        params.update({key: value for key, value in overrides.items() if value is not None})
        return {"name": name, "eval_metric": eval_metric, "params": params}

    lr = float(base["learning_rate"])
    mcw = float(base["min_child_weight"])
    lam = float(base["reg_lambda"])
    return [
        make("anchor_logloss", eval_metric="logloss"),
        make("anchor_brier", eval_metric="rmse"),
        make("slow18", learning_rate=lr / 1.45, n_estimators=10000),
        make("ultraslow18", learning_rate=lr / 2.0, n_estimators=15000),
        make(
            "moderate20",
            n_estimators=8000,
            max_depth=7,
            max_leaves=20,
            min_child_weight=mcw,
        ),
        make(
            "moderate24",
            learning_rate=lr / 1.10,
            n_estimators=9000,
            max_depth=8,
            max_leaves=24,
            min_child_weight=mcw,
            reg_lambda=lam * 1.10,
        ),
        make(
            "moderate28",
            learning_rate=lr / 1.18,
            n_estimators=10000,
            max_depth=8,
            max_leaves=28,
            min_child_weight=560.0,
            reg_lambda=lam * 1.15,
        ),
        make(
            "moderate24_diverse",
            learning_rate=lr / 1.10,
            n_estimators=9000,
            max_depth=8,
            max_leaves=24,
            min_child_weight=mcw,
            reg_lambda=lam * 1.10,
            subsample=0.90,
            colsample_bytree=0.70,
        ),
        make(
            "wide32_reg",
            learning_rate=lr / 1.25,
            n_estimators=11000,
            max_depth=8,
            max_leaves=32,
            min_child_weight=mcw,
            reg_lambda=lam * 1.20,
        ),
        make(
            "wide32_flex",
            learning_rate=lr / 1.25,
            n_estimators=11000,
            max_depth=8,
            max_leaves=32,
            min_child_weight=420.0,
        ),
        make(
            "wide48_reg",
            learning_rate=lr / 1.35,
            n_estimators=12000,
            max_depth=9,
            max_leaves=48,
            min_child_weight=mcw,
            reg_lambda=lam * 1.45,
        ),
        make(
            "wide48_flex",
            learning_rate=lr / 1.35,
            n_estimators=12000,
            max_depth=9,
            max_leaves=48,
            min_child_weight=360.0,
        ),
        make(
            "wide64_reg",
            learning_rate=lr / 1.50,
            n_estimators=14000,
            max_depth=10,
            max_leaves=64,
            min_child_weight=520.0,
            reg_lambda=lam * 1.60,
        ),
        make(
            "wide64_flex",
            learning_rate=lr / 1.50,
            n_estimators=14000,
            max_depth=10,
            max_leaves=64,
            min_child_weight=280.0,
            reg_lambda=lam * 0.75,
        ),
        make(
            "wide96_reg",
            learning_rate=lr / 1.60,
            n_estimators=15000,
            max_depth=11,
            max_leaves=96,
            min_child_weight=480.0,
            reg_lambda=lam * 1.90,
        ),
        make(
            "wide96_flex",
            learning_rate=lr / 1.60,
            n_estimators=15000,
            max_depth=11,
            max_leaves=96,
            min_child_weight=240.0,
            reg_lambda=lam * 0.70,
        ),
        make(
            "wide128_reg",
            learning_rate=lr / 1.75,
            n_estimators=17000,
            max_depth=12,
            max_leaves=128,
            min_child_weight=420.0,
            reg_lambda=lam * 2.20,
        ),
        make(
            "wide64_bin1024",
            learning_rate=lr / 1.50,
            n_estimators=14000,
            max_depth=10,
            max_leaves=64,
            min_child_weight=520.0,
            reg_lambda=lam * 1.60,
            max_bin=1024,
        ),
    ]


def paths_for(name: str, fold: int, seed_offset: int):
    stem = f"{name}_f{fold}_s{seed_offset}"
    return (
        MODEL_DIR / f"{stem}.ubj",
        PREDICTION_DIR / f"{stem}.parquet",
        OUTPUT_DIR / f"{stem}.json",
    )


def load_existing_result(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_results_table():
    records = []
    for path in OUTPUT_DIR.glob("*_f*_s*.json"):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if records:
        pd.DataFrame(records).sort_values(
            ["fold", "normalized_brier", "candidate", "seed_offset"]
        ).to_csv(RESULT_PATH, index=False)


def run_one(
    frame,
    features,
    fold,
    train_x,
    valid_x,
    train_mask,
    valid_mask,
    half_life,
    candidate,
    seed_offset,
    early_stopping_rounds,
    save_models,
    force,
):
    model_path, prediction_path, result_path = paths_for(
        candidate["name"], fold, seed_offset
    )
    existing = load_existing_result(result_path)
    if existing is not None and prediction_path.is_file() and not force:
        print(json.dumps({"status": "cached", **existing}, ensure_ascii=False), flush=True)
        return existing

    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    random_state = int(SEED + fold + seed_offset)
    model = XGBClassifier(
        **candidate["params"],
        grow_policy="lossguide",
        objective="binary:logistic",
        eval_metric=candidate["eval_metric"],
        tree_method="hist",
        device="cuda",
        random_state=random_state,
        n_jobs=6,
        early_stopping_rounds=early_stopping_rounds,
    )
    started = time.time()
    model.fit(
        train_x,
        train_y,
        sample_weight=weights,
        eval_set=[(valid_x, valid_y)],
        verbose=False,
    )
    train_elapsed_sec = time.time() - started
    predict_started = time.time()
    prediction = model.predict_proba(valid_x)[:, 1]
    predict_elapsed_sec = time.time() - predict_started
    metrics = probability_metrics(valid_y, prediction)

    model_size_bytes = None
    if save_models:
        model.save_model(str(model_path))
        model_size_bytes = model_path.stat().st_size
    result = {
        "candidate": candidate["name"],
        "feature_version": FEATURE_VERSION,
        "feature_count": len(features),
        "fold": int(fold),
        "train_through": int(fold - 1),
        "seed_offset": int(seed_offset),
        "random_state": random_state,
        "eval_metric": candidate["eval_metric"],
        "best_iteration": int(model.best_iteration),
        "tree_count": int(model.best_iteration) + 1,
        "max_depth": int(candidate["params"]["max_depth"]),
        "max_leaves": int(candidate["params"]["max_leaves"]),
        "learning_rate": float(candidate["params"]["learning_rate"]),
        "min_child_weight": float(candidate["params"]["min_child_weight"]),
        "reg_lambda": float(candidate["params"]["reg_lambda"]),
        "max_bin": int(candidate["params"]["max_bin"]),
        "half_life": float(half_life),
        "train_elapsed_sec": train_elapsed_sec,
        "predict_elapsed_sec": predict_elapsed_sec,
        "model_size_bytes": model_size_bytes,
        "model_size_mb": None if model_size_bytes is None else model_size_bytes / (1024**2),
        **metrics,
    }
    pd.DataFrame(
        {
            "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
            "season": int(fold),
            TARGET: valid_y,
            "candidate": candidate["name"],
            "seed_offset": int(seed_offset),
            "prediction": prediction.astype("float32"),
        }
    ).to_parquet(prediction_path, index=False)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_results_table()
    print(json.dumps({"status": "complete", **result}, ensure_ascii=False), flush=True)

    del model, train_y, valid_y, weights, prediction
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    trial = load_trial93()
    grid = candidate_grid(dict(trial.params))
    requested = [item.strip() for item in args.candidates.split(",") if item.strip()]
    if requested != ["all"]:
        unknown = sorted(set(requested) - {item["name"] for item in grid})
        if unknown:
            raise ValueError(f"Unknown candidates: {unknown}")
        grid = [item for item in grid if item["name"] in requested]
    folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]
    seed_offsets = [
        int(item.strip()) for item in args.seed_offsets.split(",") if item.strip()
    ]
    half_life = float(trial.params["half_life"])

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "study": STUDY_NAME,
                "trial": TRIAL_NUMBER,
                "feature_version": FEATURE_VERSION,
                "validation_rule": "train seasons strictly before validation season",
                "trackman_rule": "strictly before validation season; pitcher-season >=500",
                "half_life": half_life,
                "early_stopping_rounds": args.early_stopping_rounds,
                "candidates": grid,
                "folds": folds,
                "seed_offsets": seed_offsets,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    frame, features = load_frame_and_features()
    for fold in folds:
        train_mask, valid_mask, train_x, valid_x = encode_fold(frame, features, fold)
        print(
            json.dumps(
                {
                    "status": "fold_ready",
                    "fold": fold,
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(valid_mask.sum()),
                    "features": len(features),
                    "candidates": len(grid),
                    "seeds": seed_offsets,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        for candidate in grid:
            for seed_offset in seed_offsets:
                run_one(
                    frame,
                    features,
                    fold,
                    train_x,
                    valid_x,
                    train_mask,
                    valid_mask,
                    half_life,
                    candidate,
                    seed_offset,
                    args.early_stopping_rounds,
                    not args.no_save_models,
                    args.force,
                )
        del train_x, valid_x, train_mask, valid_mask
        gc.collect()
    save_results_table()


if __name__ == "__main__":
    main()
