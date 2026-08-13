from __future__ import annotations

import json

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool

from benchmark_game_type_experts import load_frame_and_features
from run_optuna_family import CATEGORICAL_COLUMNS, ROOT, SEED, TARGET, probability_metrics


WORK = ROOT / "experiment" / "model_optimization" / "game_type_experts"
CAT_STUDY = "cat_game_type_f_postbreak"
CAT_TRIAL = 9
XGB_TRIAL = 31
CUTOFFS = (5, 6, 7, 8)


def train_cat_sequential(frame, features, trial):
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    local = frame[features].copy()
    for column in categorical:
        local[column] = local[column].fillna("__MISSING__").astype(str)
    rows = []
    for cutoff in CUTOFFS:
        train_mask = (
            frame["season"].eq(2023)
            & frame["game_month"].le(cutoff)
            & frame["game_type"].astype(str).eq("F")
        )
        valid_mask = (
            frame["season"].eq(2023)
            & frame["game_month"].eq(cutoff + 1)
            & frame["game_type"].astype(str).eq("F")
        )
        valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
        train_pool = Pool(
            local.loc[train_mask],
            label=frame.loc[train_mask, TARGET],
            cat_features=categorical,
        )
        valid_pool = Pool(local.loc[valid_mask], label=valid_y, cat_features=categorical)
        model = CatBoostClassifier(
            **trial.params,
            loss_function="Logloss",
            eval_metric="Logloss",
            task_type="GPU",
            devices="0",
            bootstrap_type="Bayesian",
            random_seed=SEED + trial.number + cutoff,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=220,
        )
        rows.append(
            pd.DataFrame(
                {
                    "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
                    "cutoff_month": cutoff,
                    TARGET: valid_y,
                    "prediction": model.predict_proba(valid_pool)[:, 1].astype("float32"),
                    "best_iteration": int(model.get_best_iteration()),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def evaluate_grid(data):
    rows = []
    for xgb_weight in np.linspace(0.0, 0.60, 61):
        for cat_weight in np.linspace(0.0, 0.60, 61):
            if xgb_weight + cat_weight > 0.80:
                continue
            fold_nbs = []
            all_y = []
            all_prediction = []
            for _, part in data.groupby("cutoff_month"):
                global_prediction = part["global_prediction"].to_numpy("float64")
                prediction = (
                    global_prediction
                    + xgb_weight * (part["xgb_prediction"].to_numpy("float64") - global_prediction)
                    + cat_weight * (part["cat_prediction"].to_numpy("float64") - global_prediction)
                )
                y = part[TARGET].to_numpy("int8")
                metrics = probability_metrics(y, prediction)
                fold_nbs.append(metrics["normalized_brier"])
                all_y.append(y)
                all_prediction.append(prediction)
            pooled = probability_metrics(np.concatenate(all_y), np.concatenate(all_prediction))
            rows.append(
                {
                    "xgb_weight": float(xgb_weight),
                    "cat_weight": float(cat_weight),
                    "mean_fold_normalized_brier": float(np.mean(fold_nbs)),
                    "std_fold_normalized_brier": float(np.std(fold_nbs)),
                    "robust_objective": float(np.mean(fold_nbs) + 0.25 * np.std(fold_nbs)),
                    **{f"pooled_{key}": value for key, value in pooled.items()},
                }
            )
    return pd.DataFrame(rows).sort_values("robust_objective")


def main():
    frame, _, features = load_frame_and_features()
    study = optuna.load_study(
        study_name=CAT_STUDY,
        storage=f"sqlite:///{(WORK / f'{CAT_STUDY}.db').as_posix()}",
    )
    trial = next(item for item in study.trials if item.number == CAT_TRIAL)
    cat = train_cat_sequential(frame, features, trial)
    seq = pd.read_parquet(WORK / "f_sequential_oof.parquet")
    global_prediction = seq.loc[
        seq["model"].eq("global_anchor"),
        ["row_id", "cutoff_month", TARGET, "prediction"],
    ].rename(columns={"prediction": "global_prediction"})
    xgb = seq.loc[
        seq["trial"].eq(XGB_TRIAL),
        ["row_id", "cutoff_month", "prediction"],
    ].rename(columns={"prediction": "xgb_prediction"})
    data = (
        global_prediction.merge(xgb, on=["row_id", "cutoff_month"], validate="one_to_one")
        .merge(
            cat[["row_id", "cutoff_month", "prediction"]].rename(columns={"prediction": "cat_prediction"}),
            on=["row_id", "cutoff_month"],
            validate="one_to_one",
        )
    )
    grid = evaluate_grid(data)
    cat.to_parquet(WORK / "f_cat_sequential_oof.parquet", index=False)
    grid.to_csv(WORK / "f_multifamily_sequential_grid.csv", index=False)
    summary = {
        "cat_trial": CAT_TRIAL,
        "xgb_trial": XGB_TRIAL,
        "best": json.loads(grid.iloc[0].to_json()),
        "cat_iterations": cat.groupby("cutoff_month")["best_iteration"].first().to_dict(),
    }
    (WORK / "f_multifamily_sequential_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
