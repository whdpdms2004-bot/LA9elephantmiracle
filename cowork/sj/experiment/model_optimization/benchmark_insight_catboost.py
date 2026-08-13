from __future__ import annotations

import argparse
import gc
import json
import time

import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool

from benchmark_insight_features import (
    WORK_DIR,
    add_calibration_features,
    build_past_only_lookups,
)
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    SEED,
    TARGET,
    probability_metrics,
    recency_weights,
)


STUDY_NAME = "catboost_v2r200_tm500_robust"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=39)
    parser.add_argument("--folds", default="2024")
    parser.add_argument("--variants", default="INSIGHT_BASE,INSIGHT_PRIOR")
    parser.add_argument("--output-tag", default="trial39")
    return parser.parse_args()


def load_trial(number: int):
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{(WORK_DIR / f'{STUDY_NAME}.db').as_posix()}",
    )
    return next(trial for trial in study.trials if trial.number == number)


def run_one(frame, features, version, fold, trial):
    started = time.time()
    train_mask = frame["season"].lt(fold)
    valid_mask = frame["season"].eq(fold)
    cat_frame = frame[features].copy()
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    for column in categorical:
        cat_frame[column] = cat_frame[column].fillna("__MISSING__").astype(str)

    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    params = dict(trial.params)
    half_life = float(params.pop("half_life"))
    weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    train_pool = Pool(
        cat_frame.loc[train_mask],
        label=train_y,
        cat_features=categorical,
        weight=weights,
    )
    valid_pool = Pool(
        cat_frame.loc[valid_mask],
        label=valid_y,
        cat_features=categorical,
    )
    # The robust study evaluated [2023, 2024], so the 2024 fold used index 1.
    fold_index = 0 if fold == 2023 else 1 if fold == 2024 else 0
    model = CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type="GPU",
        devices="0",
        random_seed=SEED + trial.number + fold_index,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=220,
    )
    prediction = model.predict_proba(valid_pool)[:, 1]
    row = {
        "experiment": f"catboost_insight_{version.lower()}_t{trial.number}",
        "family": "catboost",
        "feature_version": version,
        "trial": trial.number,
        "fold": fold,
        "train_through": fold - 1,
        "trackman": True,
        "trackman_cutoff": fold,
        "min_trackman_season_pitches": 500,
        "feature_count": len(features),
        "best_iteration": int(model.get_best_iteration()),
        "elapsed_sec": time.time() - started,
        **probability_metrics(valid_y, prediction),
    }
    pred = pd.DataFrame(
        {
            "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
            "season": fold,
            TARGET: valid_y,
            "model": row["experiment"],
            "prediction": prediction.astype("float32"),
        }
    )
    print(json.dumps(row, ensure_ascii=False), flush=True)
    del model, train_pool, valid_pool, cat_frame, train_y, valid_y, weights, prediction
    gc.collect()
    return row, pred


def main():
    args = parse_args()
    frame, base_features = load_enhanced_frame()
    labels = pd.read_parquet(WORK_DIR / "failure_component_labels.parquet")
    lookups, audit = build_past_only_lookups(frame, labels)
    frame, _, prior_columns = add_calibration_features(frame, lookups)
    variants = {
        "INSIGHT_BASE": base_features,
        "INSIGHT_PRIOR": list(dict.fromkeys(base_features + prior_columns)),
    }
    requested = [item.strip() for item in args.variants.split(",") if item.strip()]
    folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]
    trial = load_trial(args.trial)
    results = []
    predictions = []
    for fold in folds:
        for version in requested:
            row, pred = run_one(frame, variants[version], version, fold, trial)
            results.append(row)
            predictions.append(pred)

    suffix = f"_{args.output_tag}" if args.output_tag else ""
    pd.DataFrame(results).to_csv(
        WORK_DIR / f"insight_catboost_results{suffix}.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_parquet(
        WORK_DIR / f"insight_catboost_predictions{suffix}.parquet", index=False
    )
    (WORK_DIR / f"insight_catboost_summary{suffix}.json").write_text(
        json.dumps(
            {
                "study": STUDY_NAME,
                "trial": args.trial,
                "audit_passed": all(
                    item["source_season"] is None
                    or item["source_season"] < item["target_season"]
                    for item in audit
                ),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
