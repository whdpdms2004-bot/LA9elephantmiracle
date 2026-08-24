from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool

from benchmark_game_type_experts import load_frame_and_features
from run_optuna_family import CATEGORICAL_COLUMNS, ROOT, SEED, TARGET, probability_metrics, recency_weights
from run_optuna_game_type_expert import robust_r


WORK = ROOT / "experiment" / "model_optimization" / "game_type_experts"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-type", choices=["R", "F"], required=True)
    parser.add_argument("--target-total", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args()


def suggest_params(trial, game_type):
    if game_type == "F":
        return {
            "iterations": trial.suggest_int("iterations", 500, 5000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.004, 0.08, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5.0, 500.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.05, 5.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        }
    return {
        "iterations": trial.suggest_int("iterations", 1000, 7000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
        "depth": trial.suggest_int("depth", 5, 9),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 20.0, 1000.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.05, 3.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "half_life": trial.suggest_float("half_life", 0.30, 2.0, log=True),
    }


def prepare(game_type):
    frame, _, features = load_frame_and_features()
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    local = frame[features].copy()
    for column in categorical:
        local[column] = local[column].fillna("__MISSING__").astype(str)
    if game_type == "R":
        folds = [2023, 2024]
        masks = {
            fold: (
                frame["season"].lt(fold) & frame["game_type"].astype(str).eq("R"),
                frame["season"].eq(fold) & frame["game_type"].astype(str).eq("R"),
            )
            for fold in folds
        }
    else:
        folds = [2024]
        masks = {
            2024: (
                frame["season"].eq(2023) & frame["game_type"].astype(str).eq("F"),
                frame["season"].eq(2024) & frame["game_type"].astype(str).eq("F"),
            )
        }
    return frame, local, features, categorical, folds, masks


def make_objective(game_type, frame, local, categorical, folds, masks):
    def objective(trial):
        params = suggest_params(trial, game_type)
        half_life = float(params.pop("half_life")) if game_type == "R" else None
        fold_metrics = {}
        started = time.time()
        for fold_index, fold in enumerate(folds):
            train_mask, valid_mask = masks[fold]
            if half_life is None:
                weights = np.ones(int(train_mask.sum()), dtype="float32")
            else:
                weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
            train_pool = Pool(
                local.loc[train_mask],
                label=frame.loc[train_mask, TARGET],
                weight=weights,
                cat_features=categorical,
            )
            valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
            valid_pool = Pool(
                local.loc[valid_mask], label=valid_y, cat_features=categorical
            )
            model = CatBoostClassifier(
                **params,
                loss_function="Logloss",
                eval_metric="Logloss",
                task_type="GPU",
                devices="0",
                bootstrap_type="Bayesian",
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
            fold_metrics[fold] = metrics
            trial.set_user_attr(f"fold_{fold}", metrics)
            trial.set_user_attr(f"best_iteration_{fold}", int(model.get_best_iteration()))
            del model, train_pool, valid_pool, prediction, valid_y, weights
            gc.collect()
        trial.set_user_attr("elapsed_sec", time.time() - started)
        trial.set_user_attr("game_type", game_type)
        trial.set_user_attr("training_regime", "post-break 2023 -> 2024" if game_type == "F" else "strict two-fold")
        if game_type == "F":
            return fold_metrics[2024]["normalized_brier"]
        return robust_r(fold_metrics)

    return objective


def export(study, study_name, game_type, feature_count):
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": trial.number, "objective": trial.value, **trial.params}
        for fold in (2023, 2024):
            metrics = trial.user_attrs.get(f"fold_{fold}")
            if metrics:
                row.update({f"fold_{fold}_{key}": value for key, value in metrics.items()})
                row[f"fold_{fold}_best_iteration"] = trial.user_attrs.get(f"best_iteration_{fold}")
        rows.append(row)
    leaderboard = pd.DataFrame(rows).sort_values("objective")
    leaderboard.to_csv(WORK / f"{study_name}_leaderboard.csv", index=False)
    best = {
        "study": study_name,
        "game_type": game_type,
        "feature_count": feature_count,
        "complete_trials": len(rows),
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
    }
    (WORK / f"{study_name}_best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(best, ensure_ascii=False, indent=2), flush=True)


def main():
    args = parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    study_name = f"cat_game_type_{args.game_type.lower()}_{'postbreak' if args.game_type == 'F' else 'robust'}"
    frame, local, features, categorical, folds, masks = prepare(args.game_type)
    sampler = optuna.samplers.TPESampler(
        seed=SEED + (71 if args.game_type == "R" else 83),
        multivariate=True,
        group=True,
        n_startup_trials=18,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{(WORK / f'{study_name}.db').as_posix()}",
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    complete = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    remaining = max(0, args.target_total - complete)
    print(json.dumps({"study": study_name, "complete": complete, "remaining": remaining, "folds": folds}), flush=True)
    if remaining:
        study.optimize(
            make_objective(args.game_type, frame, local, categorical, folds, masks),
            n_trials=remaining,
            timeout=args.timeout,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    export(study, study_name, args.game_type, len(features))


if __name__ == "__main__":
    main()
