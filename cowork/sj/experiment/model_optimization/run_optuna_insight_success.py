from __future__ import annotations

import argparse
import gc
import json
import time

import optuna
import pandas as pd
from xgboost import XGBClassifier

from benchmark_insight_features import (
    WORK_DIR,
    add_calibration_features,
    build_past_only_lookups,
)
from benchmark_v2_ablation import encode_fold
from run_optuna_enhanced import export_study, load_enhanced_frame
from run_optuna_family import ROOT, SEED, TARGET, probability_metrics, recency_weights


STUDY_NAME = "xgboost_insight_success_local_2024"
FEATURE_VERSION = "INSIGHT_PRIOR_SUCCESS"
FOLD = 2024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-total", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args()


def load_frame():
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
    selected = [
        column
        for column in prior_columns
        if column.startswith(("pitcher_success_", "batter_success_"))
    ]
    return frame, list(dict.fromkeys(base_features + selected))


def seed_params():
    study = optuna.load_study(
        study_name="xgboost_v2r200_tm500_local_2024",
        storage=f"sqlite:///{(WORK_DIR / 'xgboost_v2r200_tm500_local_2024.db').as_posix()}",
    )
    trial = next(item for item in study.trials if item.number == 93)
    return dict(trial.params)


def suggest_params(trial):
    return {
        "half_life": trial.suggest_float("half_life", 0.30, 0.85, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 2500, 7000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.0025, 0.012, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 180.0, 1200.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.86, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.90),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.80, 1.0),
        "gamma": trial.suggest_float("gamma", 0.03, 3.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 15.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 80.0, 1200.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [256, 512]),
        "max_leaves": trial.suggest_int("max_leaves", 12, 40, log=True),
    }


def make_objective(frame, features):
    train_mask, valid_mask, train_x, valid_x = encode_fold(frame, features, FOLD)
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")

    def objective(trial):
        started = time.time()
        params = suggest_params(trial)
        half_life = float(params.pop("half_life"))
        weights = recency_weights(frame.loc[train_mask, "season"], FOLD, half_life)
        model = XGBClassifier(
            **params,
            grow_policy="lossguide",
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda",
            random_state=SEED + FOLD,
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
        trial.set_user_attr("fold_2024", metrics)
        trial.set_user_attr("best_iteration_2024", int(model.best_iteration))
        trial.set_user_attr("elapsed_sec", time.time() - started)
        trial.set_user_attr("feature_version", FEATURE_VERSION)
        trial.set_user_attr("trackman_cutoff_rule", "season_strict")
        trial.set_user_attr("min_trackman_season_pitches", 500)
        print(
            json.dumps(
                {
                    "trial": trial.number,
                    "normalized_brier": metrics["normalized_brier"],
                    "bss": metrics["bss"],
                    "best_iteration": int(model.best_iteration),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del model, prediction, weights
        gc.collect()
        return metrics["normalized_brier"]

    return objective


def main():
    args = parse_args()
    frame, features = load_frame()
    sampler = optuna.samplers.TPESampler(
        seed=SEED + 119,
        multivariate=True,
        group=True,
        n_startup_trials=18,
    )
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{(WORK_DIR / f'{STUDY_NAME}.db').as_posix()}",
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(seed_params())
    complete = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, args.target_total - complete)
    if remaining:
        study.optimize(
            make_objective(frame, features),
            n_trials=remaining,
            timeout=args.timeout,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    export_study(study, STUDY_NAME)
    best_path = WORK_DIR / f"{STUDY_NAME}_best.json"
    best = json.loads(best_path.read_text(encoding="utf-8"))
    best["feature_version"] = FEATURE_VERSION
    best_path.write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
