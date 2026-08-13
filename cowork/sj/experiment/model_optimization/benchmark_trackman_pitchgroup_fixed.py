from __future__ import annotations

import gc
import json
import time

import pandas as pd
from xgboost import XGBClassifier

from benchmark_v2_ablation import TRIAL_NUMBER, encode_fold, load_trial
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import ROOT, SEED, TARGET, probability_metrics, recency_weights


WORK_DIR = ROOT / "experiment" / "model_optimization"


def make_feature_sets(columns):
    mix = [column for column in columns if column.startswith("tmg500_mix_")]
    metadata_tokens = (
        "_available",
        "_eligible_seasons",
        "_total_pitches",
        "_last_season",
        "_season_gap",
        "_last_n",
    )
    metadata = [column for column in columns if column.endswith(metadata_tokens)]
    mean_moments = [
        column
        for column in columns
        if any(token in column for token in ["_latest_", "_recent_"])
        and column.endswith("_mean")
    ]
    no_trend = [
        column
        for column in columns
        if "_latest_minus_recent_" not in column
    ]
    fastball = [
        column
        for column in columns
        if column.startswith("tmg500_fastball_") or column in mix
    ]
    fast_break = [
        column
        for column in columns
        if column.startswith("tmg500_fastball_")
        or column.startswith("tmg500_breaking_")
        or column.startswith("tmg500_fastball_minus_breaking_")
        or column in mix
    ]
    return {
        "BASE_NO_PITCHGROUP": [],
        "PITCHGROUP_ALL": columns,
        "PITCHGROUP_NO_TREND": no_trend,
        "PITCHGROUP_MEAN_COMPACT": list(dict.fromkeys(metadata + mean_moments + mix)),
        "PITCHGROUP_FASTBALL": fastball,
        "PITCHGROUP_FAST_BREAK": fast_break,
    }


def run_one(frame, base_features, additions, version, fold, trial):
    started = time.time()
    features = list(dict.fromkeys(base_features + additions))
    params = dict(trial.params)
    half_life = float(params.pop("half_life"))
    train_mask, valid_mask, train_x, valid_x = encode_fold(frame, features, fold)
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=SEED + fold,
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
    prediction = model.predict_proba(valid_x)[:, 1]
    experiment = f"xgboost_{version.lower()}"
    row = {
        "experiment": experiment,
        "family": "xgboost",
        "feature_version": version,
        "trial": TRIAL_NUMBER,
        "fold": fold,
        "train_through": fold - 1,
        "trackman": True,
        "trackman_cutoff": fold,
        "min_trackman_season_pitches": 500,
        "min_trackman_group_pitches": 30,
        "feature_count": len(features),
        "best_iteration": int(model.best_iteration),
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
    del model, train_x, valid_x, train_y, valid_y, weights, prediction
    gc.collect()
    return row, pred


def main():
    frame, base_features = load_enhanced_frame()
    cache = pd.read_parquet(WORK_DIR / "trackman500_pitchgroup_asof_train.parquet")
    if not frame["row_id"].equals(cache["row_id"]):
        raise RuntimeError("Pitch-group Trackman cache row order mismatch")
    columns = [column for column in cache if column not in {"row_id", "season"}]
    frame = pd.concat([frame, cache[columns]], axis=1)
    feature_sets = make_feature_sets(columns)
    trial = load_trial()
    results = []
    predictions = []
    screen = []
    for version, additions in feature_sets.items():
        row, pred = run_one(frame, base_features, additions, version, 2024, trial)
        results.append(row)
        predictions.append(pred)
        screen.append((row["normalized_brier"], version))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    for _, version in sorted(screen)[:3]:
        row, pred = run_one(
            frame, base_features, feature_sets[version], version, 2023, trial
        )
        results.append(row)
        predictions.append(pred)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    pd.DataFrame(results).to_csv(
        WORK_DIR / "trackman_pitchgroup_fixed_results.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_parquet(
        WORK_DIR / "trackman_pitchgroup_fixed_predictions.parquet", index=False
    )
    summary = {
        "base_feature_version": "V2R200_TM500_ALL",
        "trial": TRIAL_NUMBER,
        "feature_sets": feature_sets,
        "results": results,
        "strict_manifest": "trackman500_pitchgroup_manifest.json",
    }
    (WORK_DIR / "trackman_pitchgroup_fixed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
