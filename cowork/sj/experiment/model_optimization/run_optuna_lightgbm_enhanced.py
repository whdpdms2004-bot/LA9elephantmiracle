from __future__ import annotations

import argparse
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import optuna
from lightgbm import LGBMClassifier

from benchmark_v2_ablation import encode_fold
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import (
    ROOT,
    SEED,
    TARGET,
    probability_metrics,
    recency_weights,
    robust_objective,
)


WORK_DIR = ROOT / "experiment" / "model_optimization"
FEATURE_VERSION = "V2R200_TM500_ALL"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--target-total", type=int, default=None)
    parser.add_argument("--folds", default="2023,2024")
    parser.add_argument("--study-name", default="lightgbm_v2r200_tm500_robust")
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args()


def suggest_lightgbm(trial):
    max_depth = trial.suggest_categorical("max_depth", [-1, 5, 6, 7, 8, 9, 10])
    max_leaves = 384 if max_depth == -1 else min(384, 2**max_depth)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 800, 6500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.004, 0.065, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, max_leaves, log=True),
        "max_depth": max_depth,
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 8000, log=True),
        "min_sum_hessian_in_leaf": trial.suggest_float(
            "min_sum_hessian_in_leaf", 1e-3, 100.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 300.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 2.0),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
        "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
    }


def brier_eval(y_true, prediction):
    return "brier", float(np.mean((np.asarray(prediction) - np.asarray(y_true)) ** 2)), False


def make_objective(frame, features, folds):
    encoded = {fold: encode_fold(frame, features, fold) for fold in folds}

    def objective(trial):
        params = suggest_lightgbm(trial)
        half_life = trial.suggest_float("half_life", 0.20, 4.0, log=True)
        metrics_by_fold = {}
        started = time.time()
        for fold_index, fold in enumerate(folds):
            train_mask, valid_mask, train_x, valid_x = encoded[fold]
            train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
            valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
            weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
            model = LGBMClassifier(
                **params,
                objective="binary",
                metric="None",
                random_state=SEED + trial.number + fold_index,
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
                callbacks=[
                    lgb.early_stopping(220, first_metric_only=True, verbose=False)
                ],
            )
            prediction = model.predict_proba(
                valid_x, num_iteration=model.best_iteration_
            )[:, 1]
            metrics = probability_metrics(valid_y, prediction)
            metrics_by_fold[fold] = metrics
            trial.set_user_attr(f"fold_{fold}", metrics)
            trial.set_user_attr(f"best_iteration_{fold}", int(model.best_iteration_))
            del model, prediction, train_y, valid_y, weights
            gc.collect()
        trial.set_user_attr("elapsed_sec", time.time() - started)
        trial.set_user_attr("feature_version", FEATURE_VERSION)
        trial.set_user_attr("trackman_cutoff_rule", "season_strict")
        trial.set_user_attr("min_trackman_season_pitches", 500)
        if len(folds) == 1:
            return metrics_by_fold[folds[0]]["normalized_brier"]
        return robust_objective(metrics_by_fold)

    return objective


def export_study(study, study_name):
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial": trial.number,
            "objective": trial.value,
            "state": trial.state.name,
            "elapsed_sec": trial.user_attrs.get("elapsed_sec"),
            **trial.params,
        }
        for fold in [2023, 2024]:
            metrics = trial.user_attrs.get(f"fold_{fold}")
            if metrics:
                for key, value in metrics.items():
                    row[f"fold_{fold}_{key}"] = value
                row[f"fold_{fold}_best_iteration"] = trial.user_attrs.get(
                    f"best_iteration_{fold}"
                )
        rows.append(row)
    if rows:
        import pandas as pd

        leaderboard = pd.DataFrame(rows).sort_values("objective")
        leaderboard.to_csv(WORK_DIR / f"{study_name}_leaderboard.csv", index=False)
    best = {
        "study_name": study_name,
        "feature_version": FEATURE_VERSION,
        "min_trackman_season_pitches": 500,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "complete_trials": len(rows),
    }
    (WORK_DIR / f"{study_name}_best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(best, ensure_ascii=False, indent=2), flush=True)


def main():
    args = parse_args()
    folds = [int(value) for value in args.folds.split(",") if value]
    frame, features = load_enhanced_frame()
    sampler = optuna.samplers.TPESampler(
        seed=SEED, multivariate=True, group=True, n_startup_trials=20
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{(WORK_DIR / f'{args.study_name}.db').as_posix()}",
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    complete_before = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    requested_trials = (
        max(0, args.target_total - complete_before)
        if args.target_total is not None
        else args.trials
    )
    if requested_trials > 0:
        study.optimize(
            make_objective(frame, features, folds),
            n_trials=requested_trials,
            timeout=args.timeout,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    export_study(study, args.study_name)


if __name__ == "__main__":
    main()
