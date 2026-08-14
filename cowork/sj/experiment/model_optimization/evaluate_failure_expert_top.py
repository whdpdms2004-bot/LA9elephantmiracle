from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier

from benchmark_game_type_experts import encode_subset, load_frame_and_features
from run_optuna_failure_expert import HEADS, OUTPUT, metrics
from run_optuna_family import SEED, TARGET, recency_weights


GATE_FOLDS = (2023, 2024)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", choices=sorted(HEADS), required=True)
    parser.add_argument("--top-count", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    study_name = f"xgb_failure_{args.head}_robust"
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{(OUTPUT / f'{study_name}.db').as_posix()}",
    )
    complete = sorted(
        (
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ),
        key=lambda trial: trial.value,
    )
    selected = complete[: args.top_count]
    if not selected:
        raise RuntimeError(f"No complete trials in {study_name}")

    frame, features, _ = load_frame_and_features()
    labels = pd.read_parquet(
        OUTPUT.parent / "failure_component_labels.parquet",
        columns=["row_id", HEADS[args.head]],
    )
    if not frame["row_id"].equals(labels["row_id"]):
        raise RuntimeError("Failure label row order mismatch")
    label = labels[HEADS[args.head]]
    encoded = {}
    for fold in GATE_FOLDS:
        train_mask = frame["season"].lt(fold) & label.notna()
        valid_mask = frame["season"].eq(fold) & label.notna()
        train_x, valid_x = encode_subset(frame, features, train_mask, valid_mask)
        encoded[fold] = {
            "train_mask": train_mask,
            "valid_mask": valid_mask,
            "train_x": train_x,
            "valid_x": valid_x,
            "train_y": label.loc[train_mask].astype("int8").to_numpy(),
            "valid_y": label.loc[valid_mask].astype("int8").to_numpy(),
        }

    results = []
    predictions = []
    for trial in selected:
        for fold_index, fold in enumerate(GATE_FOLDS):
            item = encoded[fold]
            params = dict(trial.params)
            half_life = float(params.pop("half_life"))
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
            started = time.time()
            model.fit(
                item["train_x"],
                item["train_y"],
                sample_weight=weight,
                eval_set=[(item["valid_x"], item["valid_y"])],
                verbose=False,
            )
            prediction = model.predict_proba(item["valid_x"])[:, 1]
            result = {
                "head": args.head,
                "trial": trial.number,
                "selection_rank": selected.index(trial) + 1,
                "selection_objective": trial.value,
                "fold": fold,
                "train_through": fold - 1,
                "best_iteration": int(model.best_iteration),
                "elapsed_sec": time.time() - started,
                **metrics(item["valid_y"], prediction),
            }
            results.append(result)
            valid_mask = item["valid_mask"]
            predictions.append(
                pd.DataFrame(
                    {
                        "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
                        "season": fold,
                        TARGET: frame.loc[valid_mask, TARGET].to_numpy("int8"),
                        "component_target": item["valid_y"],
                        "head": args.head,
                        "trial": trial.number,
                        "selection_rank": selected.index(trial) + 1,
                        "prediction": prediction.astype("float32"),
                    }
                )
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
            del model, prediction, weight
            gc.collect()

    metrics_frame = pd.DataFrame(results)
    metrics_frame.to_csv(OUTPUT / f"xgb_{args.head}_gate_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        OUTPUT / f"xgb_{args.head}_gate_oof.parquet", index=False
    )
    summary = {
        "updated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "study": study_name,
        "head": args.head,
        "selection_folds": [2022, 2023],
        "gate_folds": list(GATE_FOLDS),
        "top_count": len(selected),
        "selected_trials": [trial.number for trial in selected],
        "best_gate_2024": metrics_frame.loc[
            metrics_frame["fold"].eq(2024)
        ].sort_values("normalized_brier").iloc[0].to_dict(),
    }
    (OUTPUT / f"xgb_{args.head}_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
