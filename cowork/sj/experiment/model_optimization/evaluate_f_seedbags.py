from __future__ import annotations

import json

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

from benchmark_game_type_experts import encode_subset, load_frame_and_features
from run_optuna_family import CATEGORICAL_COLUMNS, ROOT, SEED, TARGET, probability_metrics


WORK = ROOT / "experiment" / "model_optimization" / "game_type_experts"
SEED_OFFSETS = (0, 100_000, 200_000, 300_000, 400_000)


def load_trial(study_name, number):
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{(WORK / f'{study_name}.db').as_posix()}",
    )
    return next(trial for trial in study.trials if trial.number == number)


def main():
    frame, _, features = load_frame_and_features()
    train_mask = frame["season"].eq(2023) & frame["game_type"].astype(str).eq("F")
    valid_mask = frame["season"].eq(2024) & frame["game_type"].astype(str).eq("F")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    row_id = frame.loc[valid_mask, "row_id"].to_numpy()
    xgb_trial = load_trial("xgb_game_type_f_postbreak", 31)
    cat_trial = load_trial("cat_game_type_f_postbreak", 9)

    train_x, valid_x = encode_subset(frame, features, train_mask, valid_mask)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    cat_frame = frame[features].copy()
    for column in categorical:
        cat_frame[column] = cat_frame[column].fillna("__MISSING__").astype(str)
    train_pool = Pool(
        cat_frame.loc[train_mask],
        label=frame.loc[train_mask, TARGET],
        cat_features=categorical,
    )
    valid_pool = Pool(cat_frame.loc[valid_mask], label=valid_y, cat_features=categorical)

    prediction_rows = []
    metric_rows = []
    for seed_index, offset in enumerate(SEED_OFFSETS):
        xgb_seed = SEED + xgb_trial.number + offset
        cat_seed = SEED + cat_trial.number + offset
        xgb = XGBClassifier(
            **xgb_trial.params,
            grow_policy="lossguide",
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda",
            random_state=xgb_seed,
            n_jobs=6,
            early_stopping_rounds=220,
        )
        xgb.fit(
            train_x,
            frame.loc[train_mask, TARGET].to_numpy("int8"),
            eval_set=[(valid_x, valid_y)],
            verbose=False,
        )
        xgb_prediction = xgb.predict_proba(valid_x)[:, 1].astype("float32")
        cat = CatBoostClassifier(
            **cat_trial.params,
            loss_function="Logloss",
            eval_metric="Logloss",
            task_type="GPU",
            devices="0",
            bootstrap_type="Bayesian",
            random_seed=cat_seed,
            verbose=False,
            allow_writing_files=False,
        )
        cat.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=220,
        )
        cat_prediction = cat.predict_proba(valid_pool)[:, 1].astype("float32")
        for family, prediction, iteration, seed in [
            ("xgboost", xgb_prediction, int(xgb.best_iteration), xgb_seed),
            ("catboost", cat_prediction, int(cat.get_best_iteration()), cat_seed),
        ]:
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "row_id": row_id,
                        TARGET: valid_y,
                        "family": family,
                        "seed_index": seed_index,
                        "seed": seed,
                        "prediction": prediction,
                    }
                )
            )
            metric_rows.append(
                {
                    "family": family,
                    "seed_index": seed_index,
                    "seed": seed,
                    "best_iteration": iteration,
                    **probability_metrics(valid_y, prediction),
                }
            )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    global_oof = pd.read_parquet(WORK / "expert_oof_predictions.parquet")
    global_oof = global_oof.loc[
        global_oof["model"].eq("global_anchor") & global_oof["game_type"].eq("F"),
        ["row_id", "prediction"],
    ].rename(columns={"prediction": "global_prediction"})
    grid_rows = []
    for bag_size in range(1, len(SEED_OFFSETS) + 1):
        bags = {}
        for family in ("xgboost", "catboost"):
            part = predictions.loc[predictions["family"].eq(family) & predictions["seed_index"].lt(bag_size)]
            bags[family] = part.groupby("row_id", as_index=False)["prediction"].mean().rename(
                columns={"prediction": f"{family}_prediction"}
            )
        data = global_oof.merge(bags["xgboost"], on="row_id").merge(bags["catboost"], on="row_id")
        global_prediction = data["global_prediction"].to_numpy("float64")
        dx = data["xgboost_prediction"].to_numpy("float64") - global_prediction
        dc = data["catboost_prediction"].to_numpy("float64") - global_prediction
        for xgb_weight in np.linspace(0.0, 0.60, 61):
            for cat_weight in np.linspace(0.0, 0.60, 61):
                if xgb_weight + cat_weight > 0.80:
                    continue
                prediction = global_prediction + xgb_weight * dx + cat_weight * dc
                grid_rows.append(
                    {
                        "bag_size": bag_size,
                        "xgb_weight": float(xgb_weight),
                        "cat_weight": float(cat_weight),
                        **probability_metrics(valid_y, prediction),
                    }
                )
    grid = pd.DataFrame(grid_rows)
    predictions.to_parquet(WORK / "f_seedbag_oof.parquet", index=False)
    metrics.to_csv(WORK / "f_seedbag_metrics.csv", index=False)
    grid.to_csv(WORK / "f_seedbag_grid.csv", index=False)
    best = grid.sort_values("normalized_brier").groupby("bag_size", as_index=False).first()
    best.to_csv(WORK / "f_seedbag_best.csv", index=False)
    summary = {
        "seed_offsets": list(SEED_OFFSETS),
        "metrics": metrics.to_dict(orient="records"),
        "best_by_bag_size": best.to_dict(orient="records"),
    }
    (WORK / "f_seedbag_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
