from __future__ import annotations

import argparse
import gc
import time

import optuna
from xgboost import XGBClassifier

from benchmark_v2_ablation import encode_fold, load_trial
from run_optuna_enhanced import export_study, load_enhanced_frame
from run_optuna_family import ROOT, SEED, TARGET, probability_metrics, recency_weights


WORK_DIR = ROOT / "experiment" / "model_optimization"
STUDY_NAME = "xgboost_v2r200_tm500_local_2024"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-total", type=int, default=100)
    return parser.parse_args()


def fixed_params():
    return dict(load_trial().params)


def suggest_local(trial):
    params = {
        "half_life": trial.suggest_float("half_life", 0.25, 0.90, log=True),
        "grow_policy": "lossguide",
        "n_estimators": trial.suggest_int("n_estimators", 2500, 6500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.018, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 8),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 150.0, 1500.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.82, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.82),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.84, 1.0),
        "gamma": trial.suggest_float("gamma", 0.08, 3.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.10, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 30.0, 600.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [256, 512]),
        "max_leaves": trial.suggest_int("max_leaves", 16, 64, log=True),
    }
    return params


def make_objective(frame, features):
    fold = 2024
    train_mask, valid_mask, train_x, valid_x = encode_fold(frame, features, fold)
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")

    def objective(trial):
        started = time.time()
        params = suggest_local(trial)
        half_life = float(params.pop("half_life"))
        weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda",
            random_state=SEED + fold,
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
        trial.set_user_attr("feature_version", "V2R200_TM500_ALL")
        trial.set_user_attr("trackman_cutoff_rule", "season_strict")
        trial.set_user_attr("min_trackman_season_pitches", 500)
        del model, prediction, weights
        gc.collect()
        return metrics["normalized_brier"]

    return objective


def main():
    args = parse_args()
    frame, features = load_enhanced_frame()
    sampler = optuna.samplers.TPESampler(
        seed=SEED + 77, multivariate=True, group=True, n_startup_trials=15
    )
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{(WORK_DIR / f'{STUDY_NAME}.db').as_posix()}",
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(fixed_params())
    complete = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, args.target_total - complete)
    if remaining:
        study.optimize(
            make_objective(frame, features),
            n_trials=remaining,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    export_study(study, STUDY_NAME)


if __name__ == "__main__":
    main()
