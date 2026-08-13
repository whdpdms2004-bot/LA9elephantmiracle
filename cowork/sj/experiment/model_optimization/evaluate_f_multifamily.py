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
XGB_STUDY = "xgb_game_type_f_postbreak"


def train_cat(frame, features, trial):
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    local = frame[features].copy()
    for column in categorical:
        local[column] = local[column].fillna("__MISSING__").astype(str)
    train_mask = frame["season"].eq(2023) & frame["game_type"].astype(str).eq("F")
    valid_mask = frame["season"].eq(2024) & frame["game_type"].astype(str).eq("F")
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
        random_seed=SEED + trial.number,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=220,
    )
    return pd.DataFrame(
        {
            "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
            TARGET: valid_y,
            "trial": trial.number,
            "prediction": model.predict_proba(valid_pool)[:, 1].astype("float32"),
            "best_iteration": int(model.get_best_iteration()),
        }
    )


def main():
    frame, _, features = load_frame_and_features()
    study = optuna.load_study(
        study_name=CAT_STUDY,
        storage=f"sqlite:///{(WORK / f'{CAT_STUDY}.db').as_posix()}",
    )
    top_cat = sorted(
        [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE],
        key=lambda trial: trial.value,
    )[:3]
    cat = pd.concat([train_cat(frame, features, trial) for trial in top_cat], ignore_index=True)
    cat.to_parquet(WORK / "cat_game_type_f_postbreak_top_oof.parquet", index=False)

    base = pd.read_parquet(WORK / "expert_oof_predictions.parquet")
    base = base.loc[
        base["model"].eq("global_anchor") & base["game_type"].eq("F"),
        ["row_id", TARGET, "prediction"],
    ].rename(columns={"prediction": "global_prediction"})
    xgb = pd.read_parquet(WORK / "xgb_game_type_f_postbreak_top_oof.parquet")
    xgb = xgb.loc[xgb["trial"].eq(59), ["row_id", "prediction"]].rename(
        columns={"prediction": "xgb_prediction"}
    )
    rows = []
    correlations = []
    for trial, cat_part in cat.groupby("trial"):
        data = (
            base.merge(xgb, on="row_id", validate="one_to_one")
            .merge(
                cat_part[["row_id", "prediction"]].rename(columns={"prediction": "cat_prediction"}),
                on="row_id",
                validate="one_to_one",
            )
        )
        correlations.append(
            {
                "cat_trial": int(trial),
                "corr_xgb_cat_prediction": float(data["xgb_prediction"].corr(data["cat_prediction"])),
                "corr_xgb_cat_residual": float(
                    (data["xgb_prediction"] - data["global_prediction"]).corr(
                        data["cat_prediction"] - data["global_prediction"]
                    )
                ),
            }
        )
        y = data[TARGET].to_numpy("int8")
        global_prediction = data["global_prediction"].to_numpy("float64")
        dx = data["xgb_prediction"].to_numpy("float64") - global_prediction
        dc = data["cat_prediction"].to_numpy("float64") - global_prediction
        for xgb_weight in np.linspace(0.0, 0.60, 61):
            for cat_weight in np.linspace(0.0, 0.60, 61):
                if xgb_weight + cat_weight > 0.80:
                    continue
                prediction = np.clip(
                    global_prediction + xgb_weight * dx + cat_weight * dc,
                    1e-6,
                    1.0 - 1e-6,
                )
                rows.append(
                    {
                        "cat_trial": int(trial),
                        "xgb_trial": 59,
                        "xgb_weight": float(xgb_weight),
                        "cat_weight": float(cat_weight),
                        **probability_metrics(y, prediction),
                    }
                )
    grid = pd.DataFrame(rows).sort_values("normalized_brier")
    corr = pd.DataFrame(correlations)
    grid.to_csv(WORK / "f_multifamily_grid.csv", index=False)
    corr.to_csv(WORK / "f_multifamily_correlation.csv", index=False)
    summary = {
        "top_cat_trials": [trial.number for trial in top_cat],
        "xgb_trial": 59,
        "best": json.loads(grid.iloc[0].to_json()),
        "correlations": correlations,
    }
    (WORK / "f_multifamily_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
