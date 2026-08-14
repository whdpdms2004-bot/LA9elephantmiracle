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
SCHEMES = {
    "C4_MIDDLE_FIRST": "failure_class4_middle_first",
    "C4_REVERSE_FIRST": "failure_class4_reverse_first",
    "C5_OVERLAP": "failure_class5",
}
FEATURE_VARIANTS = ["V1_BASE_RECHECK", "V2_ROW_SELECTED_200"]


def run_one(frame, original, additions, label, scheme, feature_version, fold, trial):
    started = time.time()
    features = list(dict.fromkeys(original + additions))
    train_mask, valid_mask, train_x, valid_x = encode_fold(frame, features, fold)
    train_label = label.loc[train_mask]
    valid_label = label.loc[valid_mask]
    train_known = train_label.notna()
    valid_known = valid_label.notna()

    params = dict(trial.params)
    half_life = float(params.pop("half_life"))
    params["n_estimators"] = min(int(params["n_estimators"]), 1800)
    params["learning_rate"] = max(float(params["learning_rate"]), 0.012)
    num_class = int(label.max()) + 1
    full_weights = pd.Series(
        recency_weights(frame.loc[train_mask, "season"], fold, half_life),
        index=train_label.index,
    )
    model = XGBClassifier(
        **params,
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        random_state=SEED + fold + num_class,
        n_jobs=6,
        early_stopping_rounds=120,
    )
    model.fit(
        train_x.loc[train_known],
        train_label.loc[train_known].astype("int8"),
        sample_weight=full_weights.loc[train_known].to_numpy("float32"),
        eval_set=[
            (
                valid_x.loc[valid_known],
                valid_label.loc[valid_known].astype("int8"),
            )
        ],
        verbose=False,
    )
    prediction = model.predict_proba(valid_x)[:, 0]
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    experiment = f"xgboost_multiclass_{scheme.lower()}_{feature_version.lower()}"
    row = {
        "experiment": experiment,
        "family": "xgboost_multiclass",
        "feature_version": feature_version,
        "auxiliary_target": scheme,
        "trial": TRIAL_NUMBER,
        "fold": fold,
        "train_through": fold - 1,
        "trackman": False,
        "feature_count": len(features),
        "class_count": num_class,
        "train_component_coverage": float(train_known.mean()),
        "valid_component_coverage": float(valid_known.mean()),
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
    del model, train_x, valid_x, train_label, valid_label, full_weights, prediction
    gc.collect()
    return row, pred


def main():
    frame, original, feature_sets = load_frame()
    label_frame = pd.read_parquet(WORK_DIR / "failure_component_labels.parquet")
    if not frame["row_id"].equals(label_frame["row_id"]):
        raise RuntimeError("Component label row order mismatch")
    trial = load_trial()
    results = []
    predictions = []
    screen = []

    for feature_version in FEATURE_VARIANTS:
        for scheme, label_column in SCHEMES.items():
            row, pred = run_one(
                frame,
                original,
                feature_sets[feature_version],
                label_frame[label_column],
                scheme,
                feature_version,
                2024,
                trial,
            )
            results.append(row)
            predictions.append(pred)
            screen.append((row["normalized_brier"], feature_version, scheme, label_column))
            print(json.dumps(row, ensure_ascii=False), flush=True)

    selected = sorted(screen)[:3]
    for _, feature_version, scheme, label_column in selected:
        row, pred = run_one(
            frame,
            original,
            feature_sets[feature_version],
            label_frame[label_column],
            scheme,
            feature_version,
            2023,
            trial,
        )
        results.append(row)
        predictions.append(pred)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    pd.DataFrame(results).to_csv(
        WORK_DIR / "failure_multiclass_results.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_parquet(
        WORK_DIR / "failure_multiclass_predictions.parquet", index=False
    )
    summary = {
        "study": "xgboost_v1_full_2023_2024",
        "trial": TRIAL_NUMBER,
        "label_audit": "failure_component_audit.json",
        "selected_for_2023": [
            {"feature_version": item[1], "scheme": item[2]} for item in selected
        ],
        "results": results,
    }
    (WORK_DIR / "failure_multiclass_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
