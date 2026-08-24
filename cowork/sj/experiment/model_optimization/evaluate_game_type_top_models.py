from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier

from benchmark_game_type_experts import encode_subset, load_frame_and_features
from run_optuna_family import ROOT, SEED, TARGET, probability_metrics, recency_weights


WORK = ROOT / "experiment" / "model_optimization" / "game_type_experts"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-type", choices=["R", "F"], required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def train_one(frame, features, game_type, trial, fold):
    if game_type == "R":
        train_mask = frame["season"].lt(fold) & frame["game_type"].astype(str).eq("R")
        valid_mask = frame["season"].eq(fold) & frame["game_type"].astype(str).eq("R")
    else:
        train_mask = frame["season"].eq(2023) & frame["game_type"].astype(str).eq("F")
        valid_mask = frame["season"].eq(2024) & frame["game_type"].astype(str).eq("F")
    train_x, valid_x = encode_subset(frame, features, train_mask, valid_mask)
    params = dict(trial.params)
    if game_type == "R":
        half_life = float(params.pop("half_life"))
        weight = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    else:
        weight = np.ones(int(train_mask.sum()), dtype="float32")
    model = XGBClassifier(
        **params,
        grow_policy="lossguide",
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=SEED + trial.number,
        n_jobs=6,
        early_stopping_rounds=220,
    )
    model.fit(
        train_x,
        frame.loc[train_mask, TARGET].to_numpy("int8"),
        sample_weight=weight,
        eval_set=[(valid_x, frame.loc[valid_mask, TARGET].to_numpy("int8"))],
        verbose=False,
    )
    return pd.DataFrame(
        {
            "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
            "season": fold,
            "game_type": game_type,
            TARGET: frame.loc[valid_mask, TARGET].to_numpy("int8"),
            "trial": trial.number,
            "prediction": model.predict_proba(valid_x)[:, 1].astype("float32"),
            "best_iteration": int(model.best_iteration),
        }
    )


def blend_metrics(predictions, game_type):
    global_oof = pd.read_parquet(WORK / "expert_oof_predictions.parquet")
    global_oof = global_oof.loc[
        global_oof["model"].eq("global_anchor")
        & global_oof["game_type"].eq(game_type),
        ["row_id", "season", "prediction"],
    ].rename(columns={"prediction": "global_prediction"})
    rows = []
    for (trial, season), part in predictions.groupby(["trial", "season"]):
        merged = part.merge(global_oof, on=["row_id", "season"], validate="one_to_one")
        y = merged[TARGET].to_numpy("int8")
        local = merged["prediction"].to_numpy("float64")
        global_prediction = merged["global_prediction"].to_numpy("float64")
        for alpha in np.linspace(0.0, 1.0, 101):
            prediction = global_prediction + alpha * (local - global_prediction)
            rows.append(
                {
                    "trial": int(trial),
                    "season": int(season),
                    "alpha": float(alpha),
                    **probability_metrics(y, prediction),
                }
            )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    study_name = (
        "xgb_game_type_r_robust" if args.game_type == "R" else "xgb_game_type_f_postbreak"
    )
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{(WORK / f'{study_name}.db').as_posix()}",
    )
    trials = sorted(
        [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE],
        key=lambda trial: trial.value,
    )[: args.top_k]
    frame, _, features = load_frame_and_features()
    predictions = []
    folds = [2023, 2024] if args.game_type == "R" else [2024]
    for trial in trials:
        for fold in folds:
            predictions.append(train_one(frame, features, args.game_type, trial, fold))
    prediction_frame = pd.concat(predictions, ignore_index=True)
    blend = blend_metrics(prediction_frame, args.game_type)
    prediction_frame.to_parquet(WORK / f"{study_name}_top_oof.parquet", index=False)
    blend.to_csv(WORK / f"{study_name}_top_blend.csv", index=False)
    best = (
        blend.sort_values("normalized_brier")
        .groupby(["trial", "season"], as_index=False)
        .first()
        .sort_values(["season", "normalized_brier"])
    )
    best.to_csv(WORK / f"{study_name}_top_blend_best.csv", index=False)
    print(json.dumps(best.to_dict(orient="records"), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
