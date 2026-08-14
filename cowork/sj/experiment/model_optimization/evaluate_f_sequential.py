from __future__ import annotations

import json
import time

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier

from benchmark_game_type_experts import encode_subset, load_anchor_params, load_frame_and_features
from run_optuna_family import ROOT, SEED, TARGET, probability_metrics, recency_weights


WORK = ROOT / "experiment" / "model_optimization" / "game_type_experts"
F_STUDY = "xgb_game_type_f_postbreak"
CUTOFFS = (5, 6, 7, 8)
TOP_K = 5


def fit_predict(frame, features, train_mask, valid_mask, params, seed, half_life=None):
    params = dict(params)
    params.setdefault("grow_policy", "lossguide")
    train_x, valid_x = encode_subset(frame, features, train_mask, valid_mask)
    if half_life is None:
        weights = np.ones(int(train_mask.sum()), dtype="float32")
    else:
        weights = recency_weights(frame.loc[train_mask, "season"], 2023, half_life)
    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=6,
        early_stopping_rounds=220,
    )
    model.fit(
        train_x,
        frame.loc[train_mask, TARGET].to_numpy("int8"),
        sample_weight=weights,
        eval_set=[(valid_x, frame.loc[valid_mask, TARGET].to_numpy("int8"))],
        verbose=False,
    )
    return model.predict_proba(valid_x)[:, 1].astype("float32"), int(model.best_iteration)


def main():
    started = time.time()
    frame, global_features, local_features = load_frame_and_features()
    anchor, half_life = load_anchor_params()
    study = optuna.load_study(
        study_name=F_STUDY,
        storage=f"sqlite:///{(WORK / f'{F_STUDY}.db').as_posix()}",
    )
    trials = sorted(
        [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE],
        key=lambda trial: trial.value,
    )[:TOP_K]
    prediction_rows = []
    metric_rows = []
    for cutoff in CUTOFFS:
        valid_mask = (
            frame["season"].eq(2023)
            & frame["game_month"].eq(cutoff + 1)
            & frame["game_type"].astype(str).eq("F")
        )
        global_train = frame["season"].lt(2023) | (
            frame["season"].eq(2023) & frame["game_month"].le(cutoff)
        )
        local_train = (
            frame["season"].eq(2023)
            & frame["game_month"].le(cutoff)
            & frame["game_type"].astype(str).eq("F")
        )
        global_prediction, global_iteration = fit_predict(
            frame,
            global_features,
            global_train,
            valid_mask,
            anchor,
            SEED + cutoff,
            half_life,
        )
        y = frame.loc[valid_mask, TARGET].to_numpy("int8")
        row_id = frame.loc[valid_mask, "row_id"].to_numpy()
        metric_rows.append(
            {
                "cutoff_month": cutoff,
                "valid_month": cutoff + 1,
                "model": "global_anchor",
                "trial": -1,
                "best_iteration": global_iteration,
                **probability_metrics(y, global_prediction),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "row_id": row_id,
                    "cutoff_month": cutoff,
                    TARGET: y,
                    "model": "global_anchor",
                    "trial": -1,
                    "prediction": global_prediction,
                }
            )
        )
        for trial in trials:
            local_prediction, iteration = fit_predict(
                frame,
                local_features,
                local_train,
                valid_mask,
                dict(trial.params),
                SEED + cutoff + trial.number,
            )
            metric_rows.append(
                {
                    "cutoff_month": cutoff,
                    "valid_month": cutoff + 1,
                    "model": f"f_trial_{trial.number}",
                    "trial": trial.number,
                    "best_iteration": iteration,
                    **probability_metrics(y, local_prediction),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "row_id": row_id,
                        "cutoff_month": cutoff,
                        TARGET: y,
                        "model": f"f_trial_{trial.number}",
                        "trial": trial.number,
                        "prediction": local_prediction,
                    }
                )
            )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    blend_rows = []
    global_prediction = predictions.loc[
        predictions["model"].eq("global_anchor"),
        ["row_id", "cutoff_month", TARGET, "prediction"],
    ].rename(columns={"prediction": "global_prediction"})
    for trial in [trial.number for trial in trials]:
        local = predictions.loc[
            predictions["trial"].eq(trial),
            ["row_id", "cutoff_month", "prediction"],
        ].rename(columns={"prediction": "local_prediction"})
        merged = global_prediction.merge(local, on=["row_id", "cutoff_month"], validate="one_to_one")
        for alpha in np.linspace(0.0, 0.60, 61):
            fold_nbs = []
            all_y = []
            all_prediction = []
            for cutoff, part in merged.groupby("cutoff_month"):
                prediction = part["global_prediction"].to_numpy("float64") + alpha * (
                    part["local_prediction"].to_numpy("float64")
                    - part["global_prediction"].to_numpy("float64")
                )
                fold_metrics = probability_metrics(part[TARGET].to_numpy("int8"), prediction)
                fold_nbs.append(fold_metrics["normalized_brier"])
                all_y.append(part[TARGET].to_numpy("int8"))
                all_prediction.append(prediction)
            pooled = probability_metrics(np.concatenate(all_y), np.concatenate(all_prediction))
            blend_rows.append(
                {
                    "trial": trial,
                    "alpha": float(alpha),
                    "mean_fold_normalized_brier": float(np.mean(fold_nbs)),
                    "std_fold_normalized_brier": float(np.std(fold_nbs)),
                    "robust_objective": float(np.mean(fold_nbs) + 0.25 * np.std(fold_nbs)),
                    **{f"pooled_{key}": value for key, value in pooled.items()},
                }
            )
    blend = pd.DataFrame(blend_rows).sort_values("robust_objective")
    predictions.to_parquet(WORK / "f_sequential_oof.parquet", index=False)
    metrics.to_csv(WORK / "f_sequential_metrics.csv", index=False)
    blend.to_csv(WORK / "f_sequential_blend.csv", index=False)
    summary = {
        "cutoffs": list(CUTOFFS),
        "valid_months": [value + 1 for value in CUTOFFS],
        "top_trials": [trial.number for trial in trials],
        "best": json.loads(blend.iloc[0].to_json()),
        "elapsed_sec": time.time() - started,
    }
    (WORK / "f_sequential_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
