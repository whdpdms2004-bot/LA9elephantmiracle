"""C1 전용 2022/2023/2024 GPU XGBoost 탐색.

목적함수는 각 fold 이전 시즌 Target 평균으로만 계산한 last4 선형 logit
offset을 적용한 normalized Brier다. validation/test 예측 분포는 보정값 계산에
사용하지 않는다. 첫 trial은 기존 robust B0 파라미터를 그대로 재현한다.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent
SJ = HERE.parents[1]
# 2026-08-18 feature_campaign_1000 -> claude/src 이관.
# 데이터/산출물은 캠페인 폴더에 그대로 있으므로 CAMPAIGN 으로 가리킨다.
CAMPAIGN = SJ / "feature_campaign_1000"
MO = SJ / "experiment" / "model_optimization"

from evaluate_bucketed_residual import logit, sigmoid
from evaluate_train_only_season_offsets import forecast_offset
from v77_single_xgb_screen import (
    TARGET, arm_features, build_component_unique, encode,
    load_enhanced_frame, probability_metrics, recency_weights,
)


FOLDS = (2022, 2023, 2024)
FOLD_WEIGHTS = {2022: 0.15, 2023: 0.30, 2024: 0.55}
SEED = 20260818
OUT = CAMPAIGN / "outputs" / "optuna_c1"
BASE_PARAMS = MO / "xgboost_v2r200_tm500_robust_best.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", default="c1_xgb_3fold_v1")
    parser.add_argument("--early-stopping-rounds", type=int, default=180)
    return parser.parse_args()


def robust_objective(metrics_by_fold: dict[int, dict]) -> float:
    weights = np.asarray([FOLD_WEIGHTS[fold] for fold in FOLDS], dtype=float)
    ratios = np.asarray([
        metrics_by_fold[fold]["normalized_brier"] for fold in FOLDS], dtype=float)
    weights /= weights.sum()
    return 0.80 * float(weights @ ratios) + 0.20 * float(ratios.max())


def suggest(trial: optuna.Trial) -> dict:
    return {
        "half_life": trial.suggest_float("half_life", 0.20, 1.20, log=True),
        "grow_policy": "depthwise",
        "n_estimators": trial.suggest_int("n_estimators", 1800, 5200, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.018, 0.075, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 8.0, 120.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.72, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.85),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.60, 0.98),
        "gamma": trial.suggest_float("gamma", 0.5, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 3.0, 80.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 40.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256]),
    }


def prepare_folds(frame: pd.DataFrame, base_features: list[str]):
    prepared = {}
    for fold in FOLDS:
        hierarchy = build_component_unique(frame, base_features, fold)
        work = frame.copy(deep=False)
        features = arm_features(work, base_features, "C1", fold, hierarchy)
        train_mask, valid_mask, train_x, valid_x = encode(work, features, fold)
        prepared[fold] = {
            "train_x": train_x,
            "valid_x": valid_x,
            "train_y": frame.loc[train_mask, TARGET].to_numpy("int8"),
            "valid_y": frame.loc[valid_mask, TARGET].to_numpy("int8"),
            "train_season": frame.loc[train_mask, "season"].to_numpy("int16"),
            "n_features": len(features),
        }
        print(
            f"prepared fold={fold} train={len(train_x)} valid={len(valid_x)} "
            f"features={len(features)}", flush=True)
        del hierarchy, work, train_mask, valid_mask
        gc.collect()
    return prepared


def export(study: optuna.Study, path: Path) -> None:
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": trial.number, "objective": trial.value, **trial.params}
        for fold in FOLDS:
            score = trial.user_attrs.get(f"fold_{fold}", {})
            for key in ("bss_raw", "brier", "normalized_brier", "pred_mean"):
                if key in score:
                    row[f"{key}_{fold}"] = score[key]
            row[f"best_iteration_{fold}"] = trial.user_attrs.get(
                f"best_iteration_{fold}")
            row[f"offset_{fold}"] = trial.user_attrs.get(f"offset_{fold}")
        row["elapsed_sec"] = trial.user_attrs.get("elapsed_sec")
        rows.append(row)
    pd.DataFrame(rows).sort_values("objective").to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame, base_features = load_enhanced_frame()
    rates = frame.groupby("season")[TARGET].mean()
    offsets = {
        fold: forecast_offset(rates, fold, window=4, damping=1.0)
        for fold in FOLDS
    }
    prepared = prepare_folds(frame, base_features)
    del frame
    gc.collect()

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        half_life = float(params.pop("half_life"))
        scores = {}
        started = time.time()
        for fold_index, fold in enumerate(FOLDS):
            data = prepared[fold]
            weights = recency_weights(data["train_season"], fold, half_life)
            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                random_state=SEED + trial.number * 10 + fold_index,
                n_jobs=6,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            model.fit(
                data["train_x"], data["train_y"], sample_weight=weights,
                eval_set=[(data["valid_x"], data["valid_y"])], verbose=False)
            raw = model.predict_proba(data["valid_x"])[:, 1]
            prediction = sigmoid(logit(raw) + offsets[fold])
            score = probability_metrics(data["valid_y"], prediction)
            score["bss_raw"] = 100000.0 * (1.0 - score["normalized_brier"])
            scores[fold] = score
            trial.set_user_attr(f"fold_{fold}", score)
            trial.set_user_attr(f"offset_{fold}", offsets[fold])
            trial.set_user_attr(f"best_iteration_{fold}", int(model.best_iteration))
            print(
                f"trial={trial.number:02d} fold={fold} "
                f"BSS={score['bss_raw']:.2f} mean={score['pred_mean']:.5f} "
                f"iter={model.best_iteration}", flush=True)
            del model, raw, prediction, weights
            gc.collect()
        value = robust_objective(scores)
        trial.set_user_attr("elapsed_sec", time.time() - started)
        trial.set_user_attr("robust_bss", 100000.0 * (1.0 - value))
        return value

    storage = f"sqlite:///{(OUT / f'{args.study_name}.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(
        seed=SEED, n_startup_trials=6, n_ei_candidates=48,
        multivariate=True, group=True)
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="minimize",
        sampler=sampler, load_if_exists=True)
    if not study.trials:
        baseline = json.loads(BASE_PARAMS.read_text(encoding="utf-8"))["best_params"]
        study.enqueue_trial({
            key: value for key, value in baseline.items()
            if key in {
                "half_life", "n_estimators", "learning_rate", "max_depth",
                "min_child_weight", "subsample", "colsample_bytree",
                "colsample_bylevel", "gamma", "reg_alpha", "reg_lambda", "max_bin",
            }
        })
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout,
                   gc_after_trial=True, catch=(RuntimeError,))
    export(study, OUT / f"{args.study_name}_leaderboard.csv")
    best = study.best_trial
    summary = {
        "study_name": args.study_name,
        "trials_total": len(study.trials),
        "best_trial": best.number,
        "best_value": best.value,
        "best_robust_bss": 100000.0 * (1.0 - best.value),
        "best_params": best.params,
        "folds": {str(fold): best.user_attrs[f"fold_{fold}"] for fold in FOLDS},
        "offsets": offsets,
    }
    path = OUT / f"{args.study_name}_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
