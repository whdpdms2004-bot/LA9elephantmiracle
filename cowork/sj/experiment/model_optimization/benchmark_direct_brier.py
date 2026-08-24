from __future__ import annotations

import gc
import json
import time

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

from benchmark_v2_ablation import encode_fold
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import ROOT, SEED, TARGET, probability_metrics, recency_weights


WORK_DIR = ROOT / "experiment" / "model_optimization"


def load_complete(study_name):
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{(WORK_DIR / f'{study_name}.db').as_posix()}",
    )
    return [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]


def load_candidates():
    v1 = load_complete("xgboost_v1_full_2023_2024")
    fixed = next(trial for trial in v1 if trial.number == 24)
    enhanced = load_complete("xgboost_v2r200_tm500_robust")
    robust = min(enhanced, key=lambda trial: trial.value)
    recent = min(
        enhanced,
        key=lambda trial: trial.user_attrs.get("fold_2024", {}).get(
            "normalized_brier", np.inf
        ),
    )
    output = [("v1_fixed", fixed), ("enh_robust", robust), ("enh_recent", recent)]
    unique = []
    seen = set()
    for source, trial in output:
        key = (source.split("_", 1)[0], trial.number)
        if key not in seen:
            unique.append((source, trial))
            seen.add(key)
    return unique


def run_one(frame, features, source, trial, fold):
    started = time.time()
    params = dict(trial.params)
    half_life = float(params.pop("half_life"))
    train_mask, valid_mask, train_x, valid_x = encode_fold(frame, features, fold)
    train_y = frame.loc[train_mask, TARGET].to_numpy("float32")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    model = XGBRegressor(
        **params,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",
        random_state=SEED + fold + trial.number,
        n_jobs=6,
        early_stopping_rounds=220,
    )
    model.fit(
        train_x,
        train_y,
        sample_weight=weights,
        eval_set=[(valid_x, valid_y)],
        verbose=False,
    )
    raw_prediction = model.predict(valid_x)
    prediction = np.clip(raw_prediction, 1e-6, 1.0 - 1e-6)
    experiment = f"xgb_direct_brier_{source}_t{trial.number}"
    row = {
        "experiment": experiment,
        "family": "xgboost_regression",
        "feature_version": "V2R200_TM500_ALL_DIRECT_BRIER",
        "source_study": trial.study_id if hasattr(trial, "study_id") else source,
        "source": source,
        "trial": trial.number,
        "fold": fold,
        "train_through": fold - 1,
        "trackman": True,
        "trackman_cutoff": fold,
        "min_trackman_season_pitches": 500,
        "feature_count": len(features),
        "best_iteration": int(model.best_iteration),
        "raw_min": float(raw_prediction.min()),
        "raw_max": float(raw_prediction.max()),
        "clipped_fraction": float(
            np.mean((raw_prediction <= 1e-6) | (raw_prediction >= 1.0 - 1e-6))
        ),
        "elapsed_sec": time.time() - started,
        **probability_metrics(valid_y, prediction),
    }
    pred = pd.DataFrame(
        {
            "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
            "season": fold,
            TARGET: valid_y,
            "model": experiment,
            "prediction": prediction.astype("float32"),
        }
    )
    del model, train_x, valid_x, train_y, valid_y, weights, raw_prediction, prediction
    gc.collect()
    return row, pred


def main():
    frame, features = load_enhanced_frame()
    candidates = load_candidates()
    results = []
    predictions = []
    screen = []
    for source, trial in candidates:
        row, pred = run_one(frame, features, source, trial, 2024)
        results.append(row)
        predictions.append(pred)
        screen.append((row["normalized_brier"], source, trial))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    for _, source, trial in sorted(screen, key=lambda item: item[0])[:2]:
        row, pred = run_one(frame, features, source, trial, 2023)
        results.append(row)
        predictions.append(pred)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    pd.DataFrame(results).to_csv(WORK_DIR / "direct_brier_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        WORK_DIR / "direct_brier_predictions.parquet", index=False
    )
    (WORK_DIR / "direct_brier_summary.json").write_text(
        json.dumps(
            {
                "candidate_sources": [
                    {"source": source, "trial": trial.number}
                    for source, trial in candidates
                ],
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
