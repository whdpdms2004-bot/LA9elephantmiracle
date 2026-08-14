from __future__ import annotations

import gc
import json
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from benchmark_v2_ablation import (
    TRIAL_NUMBER,
    encode_fold,
    load_frame,
    load_trial,
)
from run_optuna_family import ROOT, SEED, TARGET, probability_metrics, recency_weights


WORK_DIR = ROOT / "experiment" / "model_optimization"


def enrich_trackman(frame: pd.DataFrame, tm_columns: list[str]):
    output = frame.copy()
    output["tm500_log_total_pitches"] = np.log1p(output["tm500_total_pitches"])
    output["tm500_log_last_season_n"] = np.log1p(output["tm500_last_season_n"])
    output["tm500_log_cw_total_main_n"] = np.log1p(output["cw_total_main_n"])
    output["tm500_log_cw_total_trackman_n"] = np.log1p(output["cw_total_trackman_n"])
    for column in tm_columns:
        if "_latest_" not in column:
            continue
        recent = column.replace("_latest_", "_recent_")
        if recent in output:
            output[f"{column}_minus_recent"] = output[column] - output[recent]
    quality = (
        output["cw_match_seasons"].ge(2)
        | (output["cw_mean_sim"].ge(0.90) & output["cw_min_margin"].ge(0.10))
    )
    output["tm500_high_confidence"] = quality.fillna(False).astype("int8")
    output["tm500_low_confidence"] = (
        output["tm500_available"].eq(1) & ~quality.fillna(False)
    ).astype("int8")
    return output


def feature_sets(tm_columns: list[str], enriched_columns: list[str]):
    derived = [column for column in enriched_columns if column not in tm_columns]
    compact = [
        column
        for column in tm_columns
        if column in {
            "cw_match_seasons",
            "cw_mean_sim",
            "cw_min_margin",
            "tm500_eligible_seasons",
            "tm500_total_pitches",
            "tm500_last_season",
            "tm500_season_gap",
            "tm500_last_season_n",
            "tm500_available",
            "tm500_unavailable",
        }
        or "_latest_" in column
        or "_recent_" in column
    ]
    physical_latest = [
        column
        for column in tm_columns
        if "_latest_" in column
        or column
        in {
            "cw_match_seasons",
            "cw_mean_sim",
            "cw_min_margin",
            "tm500_eligible_seasons",
            "tm500_total_pitches",
            "tm500_season_gap",
            "tm500_last_season_n",
            "tm500_available",
            "tm500_unavailable",
        }
    ]
    return {
        "TM500_ALL": list(dict.fromkeys(tm_columns + derived)),
        "TM500_COMPACT": list(dict.fromkeys(compact + derived)),
        "TM500_LATEST": list(dict.fromkeys(physical_latest + derived)),
    }


def run_one(frame, original, additions, feature_version, fold, trial):
    started = time.time()
    features = list(dict.fromkeys(original + additions))
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
    experiment = f"xgboost_{feature_version.lower()}"
    row = {
        "experiment": experiment,
        "family": "xgboost",
        "feature_version": feature_version,
        "trial": TRIAL_NUMBER,
        "fold": fold,
        "train_through": fold - 1,
        "trackman": True,
        "trackman_cutoff": fold,
        "min_trackman_season_pitches": 500,
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
    frame, original, v2_sets = load_frame()
    tm = pd.read_parquet(WORK_DIR / "trackman500_asof_train.parquet")
    if not frame["row_id"].equals(tm["row_id"]):
        raise RuntimeError("Trackman cache row order mismatch")
    tm_columns = [column for column in tm if column not in {"row_id", "season"}]
    frame = pd.concat([frame, tm[tm_columns]], axis=1)
    before = set(frame.columns)
    frame = enrich_trackman(frame, tm_columns)
    enriched_columns = tm_columns + [column for column in frame if column not in before]
    tm_sets = feature_sets(tm_columns, enriched_columns)
    trial = load_trial()
    results = []
    predictions = []
    screen = []

    base_sets = {
        "V1": v2_sets["V1_BASE_RECHECK"],
        "V2R200": v2_sets["V2_ROW_SELECTED_200"],
    }
    for base_name, base_additions in base_sets.items():
        for tm_name, tm_additions in tm_sets.items():
            version = f"{base_name}_{tm_name}"
            row, pred = run_one(
                frame,
                original,
                base_additions + tm_additions,
                version,
                2024,
                trial,
            )
            results.append(row)
            predictions.append(pred)
            screen.append((row["normalized_brier"], base_name, tm_name))
            print(json.dumps(row, ensure_ascii=False), flush=True)

    for _, base_name, tm_name in sorted(screen)[:3]:
        version = f"{base_name}_{tm_name}"
        row, pred = run_one(
            frame,
            original,
            base_sets[base_name] + tm_sets[tm_name],
            version,
            2023,
            trial,
        )
        results.append(row)
        predictions.append(pred)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    pd.DataFrame(results).to_csv(WORK_DIR / "trackman500_fixed_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        WORK_DIR / "trackman500_fixed_predictions.parquet", index=False
    )
    summary = {
        "trial": TRIAL_NUMBER,
        "strict_asof_manifest": "trackman500_asof_manifest.json",
        "feature_sets": tm_sets,
        "results": results,
    }
    (WORK_DIR / "trackman500_fixed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
