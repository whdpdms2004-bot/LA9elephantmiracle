"""최종 3WAY 설정과 동일한 고정 900회 strict-forward OOF를 만든다.

검증 fold는 predict에만 사용한다. eval_set/use_best_model/early stopping은 없다.
GPU 모델은 항상 하나씩 순차 실행한다.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
META = HERE.parent
TW = META.parent
SJ = TW.parent
CAMPAIGN = SJ / "feature_campaign_1000"
MODEL_OPT = SJ / "experiment" / "model_optimization"
CLAUDE_SRC = SJ / "claude" / "src"
LAB = SJ / "preprocess_lab"
THREE_SRC = TW / "src"
for path in (HERE, THREE_SRC, CAMPAIGN, MODEL_OPT, CLAUDE_SRC, LAB):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from guards import assert_features_clean, train_season_trend
from harness3 import SUCCESS, bss, load_labeled


OUT = META / "outputs" / "honest_oof"
PARAMS_PATH = MODEL_OPT / "catboost_v2r200_tm500_robust_best.json"
SEED = 20262844
TARGET_CONFIG = {
    "middle": {"label": "y_middle",
               "combo": ["id_frequency", "no_trackman", "temporal_cyclic"]},
    "reverse": {"label": "y_reverse",
                "combo": ["count_multiscale", "drop_ids", "trackman_quality"]},
    "outside": {"label": "y_outside",
                "combo": ["drop_ids", "no_trackman", "rate_multiscale"]},
    "mr": {"label": "y_mr",
           "combo": ["id_frequency", "no_trackman", "temporal_cyclic"]},
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from catboost import CatBoostClassifier, Pool
    from run_optuna_enhanced import load_enhanced_frame
    from run_optuna_family import CATEGORICAL_COLUMNS, recency_weights
    from v77_single_xgb_screen import (
        build_component_unique, build_component_unique_forward)
    from v80_single_catboost import make_features
    import transforms as T

    T.load_all()
    frame, enhanced = load_enhanced_frame()
    labeled = load_labeled()
    if not np.array_equal(frame["row_id"].to_numpy(), labeled["row_id"].to_numpy()):
        raise RuntimeError("label row order mismatch")
    season = frame["season"].to_numpy()
    component_ok = labeled["label_ok"].to_numpy() == 1
    base_params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(base_params.pop("half_life"))
    base_params.update({
        "iterations": 900, "learning_rate": 0.015, "depth": 8,
        "loss_function": "Logloss", "eval_metric": "Logloss",
        "random_seed": SEED, "task_type": "GPU", "devices": "0",
        "verbose": False, "allow_writing_files": False,
    })
    report = {
        "protocol": "strict forward fixed 900 iterations; no eval_set or early stopping",
        "seed": SEED, "half_life": half_life, "folds": {},
    }
    for fold in (2023, 2024):
        fold_started = time.time()
        train_mask = (season < fold) & component_ok
        valid_mask = (season == fold) & component_ok
        static = build_component_unique(frame, enhanced, fold)
        forward = build_component_unique_forward(
            frame, enhanced, fold, cache={fold: static})
        base_frame, base_features = make_features(
            frame, enhanced, fold, "F1", forward)
        for column in (SUCCESS, "season"):
            if column not in base_frame:
                base_frame[column] = frame[column].to_numpy()
        base_categorical = [column for column in CATEGORICAL_COLUMNS
                            if column in base_features]
        train_series = pd.Series(train_mask, index=frame.index)
        fold_report = {"train_rows": int(train_mask.sum()),
                       "valid_rows": int(valid_mask.sum()), "targets": {}}
        predictions = {}
        for target, config in TARGET_CONFIG.items():
            path = OUT / f"{target}_{fold}.npy"
            values = pd.to_numeric(
                labeled[config["label"]], errors="coerce").to_numpy(np.float64)
            target_train = train_mask & np.isfinite(values)
            target_valid = valid_mask & np.isfinite(values)
            if not np.array_equal(target_train, train_mask):
                raise RuntimeError(f"missing {target} train labels")
            if not np.array_equal(target_valid, valid_mask):
                raise RuntimeError(f"missing {target} valid labels")
            if path.exists():
                prediction = np.load(path).astype(np.float64)
                if len(prediction) != int(target_valid.sum()):
                    raise RuntimeError(f"cached {path.name} length mismatch")
                print(f"cached {path.name}", flush=True)
            else:
                transformed, features, categorical = T.build(
                    base_frame, base_features, base_categorical,
                    sorted(config["combo"]), train_series, fold)
                assert_features_clean(features, f"honest_oof/{target}/{fold}")
                train_frame = transformed.loc[target_train, features].copy()
                valid_frame = transformed.loc[target_valid, features].copy()
                for column in categorical:
                    train_frame[column] = (
                        train_frame[column].fillna("__MISSING__").astype(str))
                    valid_frame[column] = (
                        valid_frame[column].fillna("__MISSING__").astype(str))
                target_values = values[target_train].astype("int8")
                prior = train_season_trend(
                    target_values, season[target_train], fold)
                baseline = float(np.log(prior / (1.0 - prior)))
                weights = np.asarray(recency_weights(
                    frame.loc[target_train, "season"], fold, half_life), np.float64)
                pool_train = Pool(
                    train_frame, label=target_values, cat_features=categorical,
                    weight=weights,
                    baseline=np.full(int(target_train.sum()), baseline))
                pool_valid = Pool(
                    valid_frame, cat_features=categorical,
                    baseline=np.full(int(target_valid.sum()), baseline))
                model = CatBoostClassifier(**dict(base_params))
                started = time.time()
                print(
                    f"training {target} fold {fold}: "
                    f"{int(target_train.sum()):,} rows, {len(features)} features",
                    flush=True)
                model.fit(pool_train)
                prediction = np.clip(
                    model.predict_proba(pool_valid, thread_count=6)[:, 1],
                    1e-7, 1.0 - 1e-7)
                np.save(path, prediction)
                print(f"saved {path.name} in {time.time() - started:.1f}s", flush=True)
                del transformed, train_frame, valid_frame, weights
                del pool_train, pool_valid, model
                gc.collect()
            predictions[target] = prediction
            target_metric = bss(
                values[target_valid].astype("int8"), prediction)
            fold_report["targets"][target] = target_metric

        success = np.clip(
            1.0 - (predictions["middle"] + predictions["reverse"]
                   - predictions["mr"] + predictions["outside"]),
            1e-7, 1.0 - 1e-7)
        np.save(OUT / f"success_identity_{fold}.npy", success)
        fold_report["identity"] = bss(
            labeled.loc[valid_mask, SUCCESS].to_numpy(np.float64), success)
        fold_report["elapsed_sec"] = time.time() - fold_started
        report["folds"][str(fold)] = fold_report
        (OUT / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8")
        del base_frame, forward, static
        gc.collect()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
