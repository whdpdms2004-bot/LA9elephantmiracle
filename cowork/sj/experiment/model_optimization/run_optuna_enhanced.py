from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

from benchmark_trackman500_fixed import enrich_trackman, feature_sets
from benchmark_v2_ablation import encode_fold, load_frame
from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    ROOT,
    SEED,
    TARGET,
    probability_metrics,
    recency_weights,
    robust_objective,
    suggest_catboost,
    suggest_xgboost,
)


WORK_DIR = ROOT / "experiment" / "model_optimization"
FEATURE_VERSION = "V2R200_TM500_ALL"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["xgboost", "catboost"], required=True)
    parser.add_argument("--trials", type=int, default=160)
    parser.add_argument("--target-total", type=int, default=None)
    parser.add_argument("--folds", default="2023,2024")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args()


def load_enhanced_frame():
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
    additions = v2_sets["V2_ROW_SELECTED_200"] + tm_all
    features = list(dict.fromkeys(original + additions))
    return frame, features


def make_xgb_objective(frame, features, folds):
    encoded = {fold: encode_fold(frame, features, fold) for fold in folds}

    def objective(trial):
        params = suggest_xgboost(trial)
        half_life = trial.suggest_float("half_life", 0.20, 3.0, log=True)
        metrics_by_fold = {}
        started = time.time()
        for fold_index, fold in enumerate(folds):
            train_mask, valid_mask, train_x, valid_x = encoded[fold]
            train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
            valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
            weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                random_state=SEED + trial.number + fold_index,
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
            metrics = probability_metrics(valid_y, prediction)
            metrics_by_fold[fold] = metrics
            trial.set_user_attr(f"fold_{fold}", metrics)
            trial.set_user_attr(f"best_iteration_{fold}", int(model.best_iteration))
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


def make_cat_objective(frame, features, folds):
    cat_frame = frame[features].copy()
    for column in CATEGORICAL_COLUMNS:
        cat_frame[column] = cat_frame[column].fillna("__MISSING__").astype(str)

    def objective(trial):
        params = suggest_catboost(trial)
        half_life = trial.suggest_float("half_life", 0.25, 4.0, log=True)
        metrics_by_fold = {}
        started = time.time()
        for fold_index, fold in enumerate(folds):
            train_mask = frame["season"].lt(fold)
            valid_mask = frame["season"].eq(fold)
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
                random_seed=SEED + trial.number + fold_index,
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
            metrics = probability_metrics(valid_y, prediction)
            metrics_by_fold[fold] = metrics
            trial.set_user_attr(f"fold_{fold}", metrics)
            trial.set_user_attr(
                f"best_iteration_{fold}", int(model.get_best_iteration())
            )
            del model, prediction, train_pool, valid_pool, valid_y, weights
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
    suffix = "_".join(map(str, folds))
    study_name = args.study_name or f"{args.family}_v2r200_tm500_{suffix}"
    frame, features = load_enhanced_frame()
    print(
        json.dumps(
            {
                "study": study_name,
                "family": args.family,
                "folds": folds,
                "rows": len(frame),
                "features": len(features),
                "requested_trials": args.trials,
            }
        ),
        flush=True,
    )
    sampler = optuna.samplers.TPESampler(
        seed=SEED, multivariate=True, group=True, n_startup_trials=30
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{(WORK_DIR / f'{study_name}.db').as_posix()}",
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    objective = (
        make_xgb_objective(frame, features, folds)
        if args.family == "xgboost"
        else make_cat_objective(frame, features, folds)
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
            objective,
            n_trials=requested_trials,
            timeout=args.timeout,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    export_study(study, study_name)


if __name__ == "__main__":
    main()
