from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "experiment" / "model_optimization"
TARGET = "control_success"
SEED = 2026
FOLD_WEIGHTS = {2022: 0.15, 2023: 0.30, 2024: 0.55}

CATEGORICAL_COLUMNS = [
    "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "base_state",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
    "count_state", "runner_out_state", "handedness_matchup",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["xgboost", "catboost"])
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--folds", default="2023,2024")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--quick-rows-per-season", type=int, default=0)
    return parser.parse_args()


def add_v1_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["count_state"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    x["runner_out_state"] = x["base_state"].astype(str) + "_o" + x["outs_before"].astype(str)
    x["handedness_matchup"] = x["pitcher_hand"].astype(str) + "_" + x["batter_hand"].astype(str)
    x["score_abs"] = x["score_diff_pitcher_team"].abs()
    x["late_inning"] = (x["inning"] >= 7).astype("int8")
    x["high_leverage"] = (x["li"] >= 2.0).astype("int8")
    for col in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]:
        x[f"log1p_{col}"] = np.log1p(x[col].clip(lower=0))
    for n in [1, 3, 5]:
        x[f"pitcher_success_delta_prev{n}"] = (
            x[f"asof_pitcher_prev{n}_game_success_rate"] - x["asof_pitcher_success_rate"]
        )
        x[f"pitcher_middle_delta_prev{n}"] = (
            x[f"asof_pitcher_prev{n}_game_middle_rate"] - x["asof_pitcher_middle_rate"]
        )
    x["ball_strike_rate_sum_gap"] = (
        x["asof_pitcher_ball_rate"] + x["asof_pitcher_strike_rate"] - 1.0
    )
    return x


def load_frame(quick_rows_per_season: int) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(ROOT / "data" / "train.csv")
    if quick_rows_per_season:
        pieces = []
        for season, part in frame.groupby("season", sort=True):
            n = min(len(part), quick_rows_per_season)
            pieces.append(part.sample(n, random_state=SEED + int(season)))
        frame = pd.concat(pieces).sort_index().reset_index(drop=True)
    frame = add_v1_features(frame)
    features = [c for c in frame.columns if c not in {"row_id", TARGET}]
    return frame, features


def probability_metrics(y_true, probability):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    brier = float(np.mean((p - y) ** 2))
    rate = float(y.mean())
    reference = rate * (1.0 - rate)
    ratio = brier / reference
    return {
        "brier": brier,
        "normalized_brier": ratio,
        "bss": max(0.0, 100000.0 * (1.0 - ratio)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)),
        "target_mean": rate,
        "pred_mean": float(p.mean()),
        "mean_gap": float(p.mean() - rate),
    }


def robust_objective(fold_metrics):
    years = sorted(fold_metrics)
    weights = np.array([FOLD_WEIGHTS[y] for y in years], dtype=float)
    weights /= weights.sum()
    ratios = np.array([fold_metrics[y]["normalized_brier"] for y in years])
    weighted = float(np.dot(weights, ratios))
    return 0.80 * weighted + 0.20 * float(ratios.max())


def recency_weights(seasons, valid_year, half_life):
    age = np.maximum(valid_year - np.asarray(seasons, dtype=float), 1.0)
    weights = np.power(0.5, age / half_life)
    return (weights / weights.mean()).astype("float32")


def encode_xgboost_fold(frame, features, valid_year):
    train_mask = frame["season"].lt(valid_year)
    valid_mask = frame["season"].eq(valid_year)
    train = frame.loc[train_mask, features].copy()
    valid = frame.loc[valid_mask, features].copy()
    for col in CATEGORICAL_COLUMNS:
        values = train[col].fillna("__MISSING__").astype(str)
        mapping = {value: i for i, value in enumerate(pd.unique(values))}
        train[col] = values.map(mapping).astype("int32")
        valid[col] = (
            valid[col].fillna("__MISSING__").astype(str)
            .map(mapping).fillna(-1).astype("int32")
        )
    for col in features:
        train[col] = pd.to_numeric(train[col], errors="coerce").astype("float32")
        valid[col] = pd.to_numeric(valid[col], errors="coerce").astype("float32")
    return {
        "train_x": train,
        "train_y": frame.loc[train_mask, TARGET].to_numpy("int8"),
        "train_season": frame.loc[train_mask, "season"].to_numpy("int16"),
        "valid_x": valid,
        "valid_y": frame.loc[valid_mask, TARGET].to_numpy("int8"),
    }


def prepare_catboost_frame(frame, features):
    output = frame[features].copy()
    for col in CATEGORICAL_COLUMNS:
        output[col] = output[col].fillna("__MISSING__").astype(str)
    return output


def suggest_xgboost(trial):
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1200, 6500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.006, 0.065, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 10.0, 1500.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.62, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.70, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 30.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 300.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
        "grow_policy": grow_policy,
    }
    if grow_policy == "lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 256, log=True)
    return params


def suggest_catboost(trial):
    bootstrap = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])
    params = {
        "iterations": trial.suggest_int("iterations", 1000, 6500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.006, 0.075, log=True),
        "depth": trial.suggest_int("depth", 5, 9),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 200.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-4, 10.0, log=True),
        "bootstrap_type": bootstrap,
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "one_hot_max_size": trial.suggest_categorical("one_hot_max_size", [2, 8, 32]),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 6),
    }
    if bootstrap == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 6.0)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.60, 1.0)
    return params


def make_objective(family, frame, features, folds):
    xgb_folds = None
    cat_frame = None
    if family == "xgboost":
        xgb_folds = {year: encode_xgboost_fold(frame, features, year) for year in folds}
    else:
        cat_frame = prepare_catboost_frame(frame, features)

    def objective(trial):
        half_life = trial.suggest_float("half_life", 0.45, 8.0, log=True)
        params = suggest_xgboost(trial) if family == "xgboost" else suggest_catboost(trial)
        metrics = {}
        started = time.time()

        for fold_index, valid_year in enumerate(folds):
            if family == "xgboost":
                data = xgb_folds[valid_year]
                weights = recency_weights(data["train_season"], valid_year, half_life)
                model = XGBClassifier(
                    **params,
                    objective="binary:logistic", eval_metric="logloss",
                    tree_method="hist", device="cuda", random_state=SEED + trial.number,
                    n_jobs=6, early_stopping_rounds=220,
                )
                model.fit(
                    data["train_x"], data["train_y"], sample_weight=weights,
                    eval_set=[(data["valid_x"], data["valid_y"])], verbose=False,
                )
                prediction = model.predict_proba(data["valid_x"])[:, 1]
                target = data["valid_y"]
                best_iteration = int(model.best_iteration)
            else:
                train_mask = frame["season"].lt(valid_year)
                valid_mask = frame["season"].eq(valid_year)
                weights = recency_weights(frame.loc[train_mask, "season"], valid_year, half_life)
                train_pool = Pool(
                    cat_frame.loc[train_mask], label=frame.loc[train_mask, TARGET],
                    cat_features=CATEGORICAL_COLUMNS, weight=weights,
                )
                valid_pool = Pool(
                    cat_frame.loc[valid_mask], label=frame.loc[valid_mask, TARGET],
                    cat_features=CATEGORICAL_COLUMNS,
                )
                model = CatBoostClassifier(
                    **params,
                    loss_function="Logloss", eval_metric="Logloss",
                    task_type="GPU", devices="0", random_seed=SEED + trial.number,
                    verbose=False, allow_writing_files=False,
                )
                model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=220)
                prediction = model.predict_proba(valid_pool)[:, 1]
                target = frame.loc[valid_mask, TARGET].to_numpy("int8")
                best_iteration = int(model.get_best_iteration())
                del train_pool, valid_pool

            metrics[valid_year] = probability_metrics(target, prediction)
            trial.set_user_attr(f"fold_{valid_year}", metrics[valid_year])
            trial.set_user_attr(f"best_iteration_{valid_year}", best_iteration)
            trial.report(robust_objective(metrics), step=fold_index)
            print(
                f"trial={trial.number} family={family} fold={valid_year} "
                f"bss={metrics[valid_year]['bss']:.2f} iter={best_iteration}",
                flush=True,
            )
            del model, prediction
            gc.collect()
            if fold_index >= 1 and trial.should_prune():
                raise optuna.TrialPruned()

        value = robust_objective(metrics)
        trial.set_user_attr("elapsed_sec", time.time() - started)
        return value

    return objective


def export_leaderboard(study, destination):
    rows = []
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for trial in sorted(complete, key=lambda t: t.value):
        row = {"trial": trial.number, "objective": trial.value}
        row.update({f"param__{k}": v for k, v in trial.params.items()})
        for year in [2022, 2023, 2024]:
            metric = trial.user_attrs.get(f"fold_{year}")
            if metric:
                for key in ["brier", "normalized_brier", "bss", "auc", "logloss", "mean_gap"]:
                    row[f"{key}_{year}"] = metric[key]
                row[f"best_iteration_{year}"] = trial.user_attrs.get(f"best_iteration_{year}")
        row["elapsed_sec"] = trial.user_attrs.get("elapsed_sec")
        rows.append(row)
    pd.DataFrame(rows).to_csv(destination, index=False)


def main():
    args = parse_args()
    folds = sorted(int(v) for v in args.folds.split(",") if v)
    random.seed(SEED)
    np.random.seed(SEED)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    started = time.time()
    frame, features = load_frame(args.quick_rows_per_season)
    print(f"loaded={frame.shape} features={len(features)} family={args.family} folds={folds}", flush=True)

    study_name = args.study_name or f"{args.family}_v1_full_{'_'.join(map(str, folds))}"
    storage = f"sqlite:///{(WORK_DIR / f'{study_name}.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(
        seed=SEED, n_startup_trials=min(20, max(5, args.trials // 5)),
        n_ei_candidates=48, multivariate=True, group=True, constant_liar=True,
    )
    pruner = optuna.pruners.PatientPruner(
        optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=1), patience=1,
    )
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="minimize",
        sampler=sampler, pruner=pruner, load_if_exists=True,
    )
    study.optimize(
        make_objective(args.family, frame, features, folds),
        n_trials=args.trials, timeout=args.timeout, gc_after_trial=True,
        catch=(RuntimeError,),
    )
    export_leaderboard(study, WORK_DIR / f"{study_name}_leaderboard.csv")
    summary = {
        "study_name": study_name,
        "family": args.family,
        "folds": folds,
        "trials_total": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "elapsed_sec": time.time() - started,
    }
    (WORK_DIR / f"{study_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

