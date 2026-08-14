from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

from benchmark_game_type_experts import encode_subset, load_frame_and_features
from run_optuna_family import ROOT, SEED, recency_weights


WORK = ROOT / "experiment" / "model_optimization"
OUTPUT = WORK / "failure_experts"
LABEL_FILE = WORK / "failure_component_labels.parquet"
HEADS = {
    "middle": "middle",
    "reverse": "reverse",
    "outside": "outside_only",
}
SELECTION_FOLDS = (2022, 2023)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", choices=sorted(HEADS), required=True)
    parser.add_argument("--target-total", type=int, default=150)
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args()


def metrics(y, prediction):
    y = np.asarray(y, dtype="int8")
    p = np.clip(np.asarray(prediction, dtype="float64"), 1e-6, 1.0 - 1e-6)
    rate = float(y.mean())
    denominator = max(rate * (1.0 - rate), 1e-12)
    brier = float(np.mean((p - y) ** 2))
    normalized = brier / denominator
    return {
        "brier": brier,
        "normalized_brier": normalized,
        "bss": max(0.0, 100000.0 * (1.0 - normalized)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)),
        "target_mean": rate,
        "pred_mean": float(p.mean()),
        "mean_gap": float(p.mean() - rate),
    }


def suggest(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 1200, 8000, log=True),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.0025, 0.035, log=True
        ),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "max_leaves": trial.suggest_int("max_leaves", 8, 28),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 20.0, 1000.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.75, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.50, 0.95
        ),
        "colsample_bylevel": trial.suggest_float(
            "colsample_bylevel", 0.70, 1.0
        ),
        "gamma": trial.suggest_float("gamma", 0.01, 8.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.02, 50.0, log=True),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 20.0, 1600.0, log=True
        ),
        "max_bin": trial.suggest_categorical("max_bin", [256, 512]),
        "half_life": trial.suggest_float("half_life", 0.4, 4.0, log=True),
    }


def prepare(head):
    frame, features, _ = load_frame_and_features()
    labels = pd.read_parquet(LABEL_FILE, columns=["row_id", HEADS[head]])
    if not frame["row_id"].equals(labels["row_id"]):
        raise RuntimeError("Failure label row order mismatch")
    label = labels[HEADS[head]]
    encoded = {}
    for fold in SELECTION_FOLDS:
        train_mask = frame["season"].lt(fold) & label.notna()
        valid_mask = frame["season"].eq(fold) & label.notna()
        train_x, valid_x = encode_subset(
            frame, features, train_mask, valid_mask
        )
        encoded[fold] = {
            "train_mask": train_mask,
            "valid_mask": valid_mask,
            "train_x": train_x,
            "valid_x": valid_x,
            "train_y": label.loc[train_mask].astype("int8").to_numpy(),
            "valid_y": label.loc[valid_mask].astype("int8").to_numpy(),
        }
    return frame, features, encoded


def robust_objective(fold_metrics):
    nb22 = fold_metrics[2022]["normalized_brier"]
    nb23 = fold_metrics[2023]["normalized_brier"]
    gap22 = abs(fold_metrics[2022]["mean_gap"])
    gap23 = abs(fold_metrics[2023]["mean_gap"])
    return float(
        0.40 * nb22
        + 0.60 * nb23
        + 0.25 * abs(nb22 - nb23)
        + 0.05 * abs(gap22 - gap23)
        + 0.02 * (gap22 + gap23)
    )


def make_objective(head, frame, encoded):
    def objective(trial):
        params = suggest(trial)
        half_life = float(params.pop("half_life"))
        fold_metrics = {}
        started = time.time()
        for fold_index, fold in enumerate(SELECTION_FOLDS):
            item = encoded[fold]
            weight = recency_weights(
                frame.loc[item["train_mask"], "season"], fold, half_life
            )
            model = XGBClassifier(
                **params,
                grow_policy="lossguide",
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                random_state=SEED + trial.number * 7 + fold_index,
                n_jobs=6,
                early_stopping_rounds=180,
            )
            model.fit(
                item["train_x"],
                item["train_y"],
                sample_weight=weight,
                eval_set=[(item["valid_x"], item["valid_y"])],
                verbose=False,
            )
            prediction = model.predict_proba(item["valid_x"])[:, 1]
            fold_metrics[fold] = metrics(item["valid_y"], prediction)
            trial.set_user_attr(f"fold_{fold}", fold_metrics[fold])
            trial.set_user_attr(
                f"best_iteration_{fold}", int(model.best_iteration)
            )
            trial.report(fold_metrics[fold]["normalized_brier"], fold_index)
            del model, prediction, weight
            gc.collect()
            if fold_index == 0 and trial.should_prune():
                raise optuna.TrialPruned()
        objective_value = robust_objective(fold_metrics)
        trial.set_user_attr("robust_objective", objective_value)
        trial.set_user_attr("elapsed_sec", time.time() - started)
        trial.set_user_attr("head", head)
        trial.set_user_attr(
            "trackman_rule", "strictly before validation; pitcher-season >=500"
        )
        return objective_value

    return objective


def export(study, head, feature_count):
    rows = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "state": trial.state.name,
            "objective": trial.value,
            **trial.params,
        }
        for fold in SELECTION_FOLDS:
            fold_metrics = trial.user_attrs.get(f"fold_{fold}")
            if fold_metrics:
                row.update(
                    {
                        f"fold_{fold}_{key}": value
                        for key, value in fold_metrics.items()
                    }
                )
                row[f"fold_{fold}_best_iteration"] = trial.user_attrs.get(
                    f"best_iteration_{fold}"
                )
        row["elapsed_sec"] = trial.user_attrs.get("elapsed_sec")
        rows.append(row)
    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            ["state", "objective"], na_position="last"
        )
    leaderboard.to_csv(OUTPUT / f"xgb_{head}_leaderboard.csv", index=False)
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    best = min(complete, key=lambda trial: trial.value) if complete else None
    status = {
        "updated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "study": f"xgb_failure_{head}_robust",
        "head": head,
        "feature_count": feature_count,
        "selection_folds": list(SELECTION_FOLDS),
        "attempted_trials": len(study.trials),
        "complete_trials": len(complete),
        "pruned_trials": sum(
            trial.state == optuna.trial.TrialState.PRUNED
            for trial in study.trials
        ),
        "best_trial": best.number if best else None,
        "best_value": best.value if best else None,
        "best_params": best.params if best else None,
        "best_user_attrs": best.user_attrs if best else None,
    }
    (OUTPUT / f"xgb_{head}_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return status


def main():
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, features, encoded = prepare(args.head)
    study_name = f"xgb_failure_{args.head}_robust"
    sampler = optuna.samplers.TPESampler(
        seed=SEED + {"middle": 101, "reverse": 103, "outside": 107}[args.head],
        multivariate=True,
        group=True,
        n_startup_trials=24,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=30, n_warmup_steps=0, interval_steps=1
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{(OUTPUT / f'{study_name}.db').as_posix()}",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    attempted = sum(
        trial.state
        in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
        for trial in study.trials
    )
    remaining = max(0, args.target_total - attempted)

    def callback(current_study, _):
        status = export(current_study, args.head, len(features))
        print(json.dumps(status, ensure_ascii=False), flush=True)

    if remaining:
        study.optimize(
            make_objective(args.head, frame, encoded),
            n_trials=remaining,
            timeout=args.timeout,
            callbacks=[callback],
            gc_after_trial=True,
            show_progress_bar=False,
        )
    status = export(study, args.head, len(features))
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
