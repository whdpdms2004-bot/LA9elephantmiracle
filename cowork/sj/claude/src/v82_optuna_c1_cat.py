"""최근 fold(2024)에서 C1 CatBoost를 BrierScore 기준으로 GPU 탐색한다.

탐색 후 상위 설정은 반드시 2022/2023에 별도 confirm한다. offset은 fold 이전
시즌 Target 평균의 last4 선형 추세 25%만 사용하며 예측 분포는 참조하지 않는다.
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
from catboost import CatBoostClassifier, Pool


HERE = Path(__file__).resolve().parent
SJ = HERE.parents[1]
# 2026-08-18 feature_campaign_1000 -> claude/src 이관.
# 데이터/산출물은 캠페인 폴더에 그대로 있으므로 CAMPAIGN 으로 가리킨다.
CAMPAIGN = SJ / "feature_campaign_1000"
MO = SJ / "experiment" / "model_optimization"

from evaluate_bucketed_residual import logit, sigmoid
from evaluate_train_only_season_offsets import forecast_offset
from v77_single_xgb_screen import (
    CATEGORICAL_COLUMNS, TARGET, arm_features, build_component_unique,
    load_enhanced_frame, probability_metrics, recency_weights,
)


FOLD = 2024
SEED = 20260818
OUT = CAMPAIGN / "outputs" / "optuna_c1_cat"
BASE_PARAMS = MO / "catboost_v2r200_tm500_robust_best.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--study-name", default="c1_cat_val2024_v1")
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args()


def suggest(trial: optuna.Trial) -> dict:
    return {
        "half_life": trial.suggest_float("half_life", 0.70, 3.0, log=True),
        "iterations": trial.suggest_int("iterations", 2200, 6200, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.030, log=True),
        "depth": trial.suggest_int("depth", 7, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 30.0, 350.0, log=True),
        "random_strength": trial.suggest_float(
            "random_strength", 1e-5, 0.20, log=True),
        "border_count": trial.suggest_categorical("border_count", [128, 254]),
        "one_hot_max_size": trial.suggest_categorical("one_hot_max_size", [8, 32]),
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations", 1, 4),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 5.0),
        "bootstrap_type": "Bayesian",
    }


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame, base_features = load_enhanced_frame()
    hierarchy = build_component_unique(frame, base_features, FOLD)
    work = frame.copy(deep=False)
    features = arm_features(work, base_features, "C1", FOLD, hierarchy)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    model_frame = work[features].copy()
    for column in categorical:
        model_frame[column] = model_frame[column].fillna("__MISSING__").astype(str)
    train_mask = frame["season"].lt(FOLD)
    valid_mask = frame["season"].eq(FOLD)
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    train_season = frame.loc[train_mask, "season"].to_numpy("int16")
    rates = frame.groupby("season")[TARGET].mean()
    offset = forecast_offset(rates, FOLD, window=4, damping=0.25)
    del frame, work, hierarchy
    gc.collect()
    print(
        f"prepared fold={FOLD} train={train_mask.sum()} valid={valid_mask.sum()} "
        f"features={len(features)} offset={offset:+.8f}", flush=True)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        half_life = float(params.pop("half_life"))
        weights = recency_weights(train_season, FOLD, half_life)
        train_pool = Pool(
            model_frame.loc[train_mask], label=train_y,
            cat_features=categorical, weight=weights)
        valid_pool = Pool(
            model_frame.loc[valid_mask], label=valid_y,
            cat_features=categorical)
        started = time.time()
        model = CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="BrierScore",
            task_type="GPU",
            devices="0",
            random_seed=SEED + trial.number,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            train_pool, eval_set=valid_pool, use_best_model=True,
            early_stopping_rounds=240)
        raw = model.predict_proba(valid_pool)[:, 1]
        prediction = sigmoid(logit(raw) + offset)
        score = probability_metrics(valid_y, prediction)
        bss_raw = 100000.0 * (1.0 - score["normalized_brier"])
        score["bss_raw"] = bss_raw
        trial.set_user_attr("fold_2024", score)
        trial.set_user_attr("best_iteration", int(model.get_best_iteration()))
        trial.set_user_attr("elapsed_sec", time.time() - started)
        print(
            f"trial={trial.number:02d} BSS={bss_raw:.2f} "
            f"mean={score['pred_mean']:.5f} iter={model.get_best_iteration()} "
            f"t={time.time()-started:.1f}s", flush=True)
        del model, raw, prediction, train_pool, valid_pool, weights
        gc.collect()
        return score["normalized_brier"]

    storage = f"sqlite:///{(OUT / f'{args.study_name}.db').as_posix()}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=SEED, n_startup_trials=4, n_ei_candidates=32,
            multivariate=True, group=True),
        load_if_exists=True,
    )
    if not study.trials:
        baseline = json.loads(BASE_PARAMS.read_text(encoding="utf-8"))["best_params"]
        study.enqueue_trial({
            key: value for key, value in baseline.items()
            if key in {
                "half_life", "iterations", "learning_rate", "depth",
                "l2_leaf_reg", "random_strength", "border_count",
                "one_hot_max_size", "leaf_estimation_iterations",
                "bagging_temperature",
            }
        })
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout,
                   gc_after_trial=True, catch=(RuntimeError,))

    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        score = trial.user_attrs["fold_2024"]
        rows.append({
            "trial": trial.number,
            "objective": trial.value,
            "bss_raw": score["bss_raw"],
            "brier": score["brier"],
            "pred_mean": score["pred_mean"],
            "best_iteration": trial.user_attrs["best_iteration"],
            "elapsed_sec": trial.user_attrs["elapsed_sec"],
            **trial.params,
        })
    pd.DataFrame(rows).sort_values("objective").to_csv(
        OUT / f"{args.study_name}_leaderboard.csv", index=False)
    best = study.best_trial
    summary = {
        "study_name": args.study_name,
        "trials_total": len(study.trials),
        "best_trial": best.number,
        "best_params": best.params,
        "fold_2024": best.user_attrs["fold_2024"],
        "best_iteration": best.user_attrs["best_iteration"],
        "offset": offset,
        "selection_gate": "confirm on untouched Val2022 and Val2023 before adoption",
    }
    path = OUT / f"{args.study_name}_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
