"""strict F1 CatBoost 설정을 최근 fold 2024의 Brier로 GPU 탐색한다.

seed는 모든 trial에서 고정해 설정 효과만 비교한다. 선택된 설정은 2022/2023에서
별도 confirm하며, 2024 우선 recent-weighted 목적과 세 fold 양수 게이트를 적용한다.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool


HERE = Path(__file__).resolve().parent
SJ = HERE.parent
MODEL_OPT = SJ / "experiment" / "model_optimization"
sys.path.insert(0, str(MODEL_OPT))
sys.path.insert(0, str(HERE))

from evaluate_bucketed_residual import logit, sigmoid
from evaluate_train_only_season_offsets import forecast_offset
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    TARGET,
    probability_metrics,
    recency_weights,
)
from v77_single_xgb_screen import (
    build_component_unique,
    build_component_unique_forward,
)
from v80_single_catboost import make_features, raw_bss


FOLD = 2024
SEED = 20262844
OUTPUT = HERE / "outputs" / "optuna_f1_cat_2024"
BASE_PARAMS = MODEL_OPT / "catboost_v2r200_tm500_robust_best.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--study-name", default="f1_cat_val2024_v1")
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args()


def suggest(trial: optuna.Trial) -> dict:
    return {
        "half_life": trial.suggest_float("half_life", 0.90, 2.80, log=True),
        "iterations": trial.suggest_int("iterations", 1800, 5000, log=True),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.0055, 0.018, log=True),
        "depth": trial.suggest_int("depth", 7, 9),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", 60.0, 280.0, log=True),
        "random_strength": trial.suggest_float(
            "random_strength", 1e-5, 0.08, log=True),
        "border_count": trial.suggest_categorical("border_count", [128, 254]),
        "one_hot_max_size": trial.suggest_categorical(
            "one_hot_max_size", [8, 32]),
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations", 1, 2),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", 1.0, 4.0),
        "bootstrap_type": "Bayesian",
    }


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, base_features = load_enhanced_frame()
    static = build_component_unique(frame, base_features, FOLD)
    forward = build_component_unique_forward(
        frame, base_features, FOLD, cache={FOLD: static})
    work, features = make_features(
        frame, base_features, FOLD, "F1", forward)
    categorical = [
        column for column in CATEGORICAL_COLUMNS if column in features]
    model_frame = work[features].copy()
    for column in categorical:
        model_frame[column] = (
            model_frame[column].fillna("__MISSING__").astype(str))
    train_mask = frame["season"].lt(FOLD)
    valid_mask = frame["season"].eq(FOLD)
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    train_season = frame.loc[train_mask, "season"].to_numpy("int16")
    rates = frame.groupby("season")[TARGET].mean()
    offset = forecast_offset(rates, FOLD, window=None, damping=0.25)
    del frame, work, forward, static
    gc.collect()
    print(
        f"prepared F1 fold={FOLD} train={int(train_mask.sum())} "
        f"valid={int(valid_mask.sum())} features={len(features)} "
        f"offset={offset:+.8f} seed={SEED}", flush=True)

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
            eval_metric="Logloss",
            task_type="GPU",
            devices="0",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=240,
        )
        raw_prediction = model.predict_proba(valid_pool)[:, 1]
        prediction = sigmoid(logit(raw_prediction) + offset)
        score = probability_metrics(valid_y, prediction)
        bss = raw_bss(score)
        elapsed = time.time() - started
        trial.set_user_attr("fold_2024", {**score, "bss_raw": bss})
        trial.set_user_attr("best_iteration", int(model.get_best_iteration()))
        trial.set_user_attr("elapsed_sec", elapsed)
        print(
            f"trial={trial.number:02d} BSS={bss:.3f} "
            f"mean={score['pred_mean']:.6f} "
            f"iter={model.get_best_iteration()} t={elapsed:.1f}s",
            flush=True,
        )
        del model, raw_prediction, prediction, train_pool, valid_pool, weights
        gc.collect()
        return score["normalized_brier"]

    storage = f"sqlite:///{(OUTPUT / f'{args.study_name}.db').as_posix()}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=SEED,
            n_startup_trials=5,
            n_ei_candidates=48,
            multivariate=True,
            group=True,
        ),
        load_if_exists=True,
    )
    if not study.trials:
        baseline = json.loads(
            BASE_PARAMS.read_text(encoding="utf-8"))["best_params"]
        study.enqueue_trial({
            key: value for key, value in baseline.items()
            if key in {
                "half_life", "iterations", "learning_rate", "depth",
                "l2_leaf_reg", "random_strength", "border_count",
                "one_hot_max_size", "leaf_estimation_iterations",
                "bagging_temperature",
            }
        })
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
        catch=(RuntimeError,),
    )

    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        score = trial.user_attrs["fold_2024"]
        rows.append({
            "trial": trial.number,
            "objective": trial.value,
            "bss_adjusted": score["bss_raw"],
            "brier": score["brier"],
            "pred_mean": score["pred_mean"],
            "best_iteration": trial.user_attrs["best_iteration"],
            "elapsed_sec": trial.user_attrs["elapsed_sec"],
            **trial.params,
        })
    leaderboard = pd.DataFrame(rows).sort_values("objective")
    leaderboard.to_csv(
        OUTPUT / f"{args.study_name}_leaderboard.csv", index=False)
    best = study.best_trial
    summary = {
        "study_name": args.study_name,
        "trials_total": len(study.trials),
        "fixed_seed": SEED,
        "best_trial": best.number,
        "best_params": best.params,
        "fold_2024": best.user_attrs["fold_2024"],
        "best_iteration": best.user_attrs["best_iteration"],
        "offset": offset,
        "selection_rule": (
            "2024 screen, then 1:2:4 recent-weighted confirm on 2022/2023/2024; "
            "all folds raw BSS > 0"),
    }
    path = OUTPUT / f"{args.study_name}_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + leaderboard.head(8).round(6).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
