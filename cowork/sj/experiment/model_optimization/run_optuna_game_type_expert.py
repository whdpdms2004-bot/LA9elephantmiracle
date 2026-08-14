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

from benchmark_game_type_experts import encode_subset, load_frame_and_features
from run_optuna_family import ROOT, SEED, TARGET, probability_metrics, recency_weights


WORK_DIR = ROOT / "experiment" / "model_optimization"
OUTPUT_DIR = WORK_DIR / "game_type_experts"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-type", choices=["R", "F"], required=True)
    parser.add_argument("--target-total", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args()


def suggest_r(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 2500, 10000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.0018, 0.018, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "max_leaves": trial.suggest_int("max_leaves", 10, 28),
        "min_child_weight": trial.suggest_float("min_child_weight", 120.0, 1400.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.82, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.92),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.75, 1.0),
        "gamma": trial.suggest_float("gamma", 0.02, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.03, 30.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 40.0, 1600.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [256, 512]),
        "half_life": trial.suggest_float("half_life", 0.30, 2.0, log=True),
    }


def suggest_f(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 500, 7000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.035, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "max_leaves": trial.suggest_int("max_leaves", 4, 20),
        "min_child_weight": trial.suggest_float("min_child_weight", 3.0, 250.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.72, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.65, 1.0),
        "gamma": trial.suggest_float("gamma", 0.01, 8.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.02, 50.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 1200.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
    }


def robust_r(metrics):
    nb23 = metrics[2023]["normalized_brier"]
    nb24 = metrics[2024]["normalized_brier"]
    gap23 = abs(metrics[2023]["mean_gap"])
    gap24 = abs(metrics[2024]["mean_gap"])
    return float(0.45 * nb23 + 0.55 * nb24 + 0.25 * abs(nb23 - nb24) + 0.05 * abs(gap23 - gap24))


def prepare(game_type):
    frame, _, features = load_frame_and_features()
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
    encoded = {}
    for fold, (train_mask, valid_mask) in masks.items():
        train_x, valid_x = encode_subset(frame, features, train_mask, valid_mask)
        encoded[fold] = {
            "train_mask": train_mask,
            "valid_mask": valid_mask,
            "train_x": train_x,
            "valid_x": valid_x,
            "train_y": frame.loc[train_mask, TARGET].to_numpy("int8"),
            "valid_y": frame.loc[valid_mask, TARGET].to_numpy("int8"),
        }
    return frame, features, folds, encoded


def make_objective(game_type, frame, folds, encoded):
    def objective(trial):
        started = time.time()
        params = suggest_r(trial) if game_type == "R" else suggest_f(trial)
        half_life = float(params.pop("half_life")) if game_type == "R" else None
        fold_metrics = {}
        for fold_index, fold in enumerate(folds):
            item = encoded[fold]
            if game_type == "R":
                weight = recency_weights(
                    frame.loc[item["train_mask"], "season"], fold, half_life
                )
            else:
                weight = np.ones(len(item["train_y"]), dtype="float32")
            model = XGBClassifier(
                **params,
                grow_policy="lossguide",
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                random_state=SEED + trial.number + fold_index,
                n_jobs=6,
                early_stopping_rounds=220,
            )
            model.fit(
                item["train_x"],
                item["train_y"],
                sample_weight=weight,
                eval_set=[(item["valid_x"], item["valid_y"])],
                verbose=False,
            )
            prediction = model.predict_proba(item["valid_x"])[:, 1]
            metrics = probability_metrics(item["valid_y"], prediction)
            fold_metrics[fold] = metrics
            trial.set_user_attr(f"fold_{fold}", metrics)
            trial.set_user_attr(f"best_iteration_{fold}", int(model.best_iteration))
            del model, prediction, weight
            gc.collect()
        trial.set_user_attr("elapsed_sec", time.time() - started)
        trial.set_user_attr("game_type", game_type)
        trial.set_user_attr("feature_version", "INSIGHT_SUCCESS_ADJUSTED_LOCAL")
        trial.set_user_attr("trackman_rule", "strictly before validation; pitcher-season >=500")
        if game_type == "F":
            trial.set_user_attr("training_regime", "post-break 2023 only -> Val2024")
            return fold_metrics[2024]["normalized_brier"]
        return robust_r(fold_metrics)

    return objective


def export(study, study_name, game_type, features):
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial": trial.number,
            "objective": trial.value,
            "elapsed_sec": trial.user_attrs.get("elapsed_sec"),
            **trial.params,
        }
        for fold in (2023, 2024):
            metrics = trial.user_attrs.get(f"fold_{fold}")
            if metrics:
                row.update({f"fold_{fold}_{key}": value for key, value in metrics.items()})
                row[f"fold_{fold}_best_iteration"] = trial.user_attrs.get(f"best_iteration_{fold}")
        rows.append(row)
    leaderboard = pd.DataFrame(rows).sort_values("objective")
    leaderboard.to_csv(OUTPUT_DIR / f"{study_name}_leaderboard.csv", index=False)
    best = {
        "study": study_name,
        "game_type": game_type,
        "feature_version": "INSIGHT_SUCCESS_ADJUSTED_LOCAL",
        "feature_count": len(features),
        "complete_trials": len(rows),
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
    }
    (OUTPUT_DIR / f"{study_name}_best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(best, ensure_ascii=False, indent=2), flush=True)


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study_name = (
        "xgb_game_type_r_robust" if args.game_type == "R" else "xgb_game_type_f_postbreak"
    )
    frame, features, folds, encoded = prepare(args.game_type)
    sampler = optuna.samplers.TPESampler(
        seed=SEED + (31 if args.game_type == "R" else 47),
        multivariate=True,
        group=True,
        n_startup_trials=24,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{(OUTPUT_DIR / f'{study_name}.db').as_posix()}",
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    complete = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, args.target_total - complete)
    print(
        json.dumps(
            {
                "study": study_name,
                "game_type": args.game_type,
                "features": len(features),
                "folds": folds,
                "complete": complete,
                "remaining": remaining,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if remaining:
        study.optimize(
            make_objective(args.game_type, frame, folds, encoded),
            n_trials=remaining,
            timeout=args.timeout,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    export(study, study_name, args.game_type, features)


if __name__ == "__main__":
    main()
