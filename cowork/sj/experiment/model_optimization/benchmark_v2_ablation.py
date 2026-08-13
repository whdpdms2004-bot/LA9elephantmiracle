from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier

from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    ROOT,
    SEED,
    TARGET,
    probability_metrics,
    recency_weights,
)


WORK_DIR = ROOT / "experiment" / "model_optimization"
STUDY_NAME = "xgboost_v1_full_2023_2024"
TRIAL_NUMBER = 24


def load_trial():
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{(WORK_DIR / f'{STUDY_NAME}.db').as_posix()}",
    )
    return next(item for item in study.trials if item.number == TRIAL_NUMBER)


def select_feature_sets(cache_columns: list[str], manifest: dict) -> dict[str, list[str]]:
    row = manifest["row_feature_columns"]
    temporal = manifest["temporal_feature_columns"]
    v1 = row[: row.index("pitcher_success_is_missing")]
    row_extra = [column for column in row if column not in v1]

    selected_200 = [
        column
        for column in row_extra
        if column.endswith("_is_missing")
        or column.endswith("_smoothed_200")
        or column.endswith("_reliability_200")
        or "_n_eq_0" in column
        or "_n_le_" in column
        or column.startswith("pitcher_recent_")
        or column in {"pitcher_failure_rate_sum", "pitcher_control_component_gap"}
    ]
    selected_500 = [
        column
        for column in row_extra
        if column.endswith("_is_missing")
        or column.endswith("_smoothed_500")
        or column.endswith("_reliability_500")
        or "_n_eq_0" in column
        or "_n_le_" in column
        or column.startswith("pitcher_recent_")
        or column in {"pitcher_failure_rate_sum", "pitcher_control_component_gap"}
    ]
    te_global = [column for column in temporal if column.startswith("te_global_")]

    def te_groups(names: set[str], fields: set[str] | None = None):
        output = list(te_global)
        for name in names:
            prefix = f"te_{name}_"
            candidates = [column for column in temporal if column.startswith(prefix)]
            if fields is not None:
                candidates = [
                    column
                    for column in candidates
                    if any(column.endswith(f"_{field}") for field in fields)
                ]
            output.extend(candidates)
        return list(dict.fromkeys(output))

    coarse = {
        "pitcher_team",
        "batter_team",
        "game_type",
        "count",
        "hand_matchup",
        "pitcher_team_game_type",
    }
    player = {"pitcher", "batter"}
    pitcher_context = {
        "pitcher",
        "pitcher_batter_hand",
        "pitcher_count",
        "pitcher_game_type",
    }
    sets = {
        "V1_BASE_RECHECK": v1,
        "V2_ROW_ALL": row,
        "V2_ROW_SELECTED_200": v1 + selected_200,
        "V2_ROW_SELECTED_500": v1 + selected_500,
        "V2_TE_GLOBAL": v1 + te_global,
        "V2_TE_COARSE": v1 + te_groups(coarse),
        "V2_TE_PLAYER": v1 + te_groups(player),
        "V2_TE_PLAYER_ALL": v1 + te_groups(player, {"all", "log_all_n"}),
        "V2_TE_PITCHER_CONTEXT": v1 + te_groups(pitcher_context),
        "V2_TE_ALL": v1 + temporal,
        "V2_SELECTED200_TE_COARSE": v1 + selected_200 + te_groups(coarse),
        "V2_SELECTED200_TE_PLAYER": v1 + selected_200 + te_groups(player),
    }
    for name, columns in sets.items():
        sets[name] = list(dict.fromkeys(column for column in columns if column in cache_columns))
    return sets


def load_frame():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    cache = pd.read_parquet(WORK_DIR / "v2_temporal_train.parquet")
    manifest = json.loads((WORK_DIR / "v2_temporal_manifest.json").read_text(encoding="utf-8"))
    if len(train) != len(cache) or not train["row_id"].equals(cache["row_id"]):
        raise RuntimeError("V2 cache row order mismatch")
    added = [column for column in cache if column not in {"row_id", "season"}]
    frame = pd.concat([train, cache[added]], axis=1)
    sets = select_feature_sets(added, manifest)
    original = [column for column in train if column not in {"row_id", TARGET}]
    return frame, original, sets


def encode_fold(frame: pd.DataFrame, features: list[str], fold: int):
    train_mask = frame["season"].lt(fold)
    valid_mask = frame["season"].eq(fold)
    train_x = frame.loc[train_mask, features].copy()
    valid_x = frame.loc[valid_mask, features].copy()
    for column in CATEGORICAL_COLUMNS:
        values = train_x[column].fillna("__MISSING__").astype(str)
        mapping = {value: index for index, value in enumerate(pd.unique(values))}
        train_x[column] = values.map(mapping).astype("int32")
        valid_x[column] = (
            valid_x[column]
            .fillna("__MISSING__")
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype("int32")
        )
    train_x = train_x.apply(pd.to_numeric, errors="coerce").astype("float32")
    valid_x = valid_x.apply(pd.to_numeric, errors="coerce").astype("float32")
    return train_mask, valid_mask, train_x, valid_x


def run_one(frame, features, feature_version, fold, trial):
    started = time.time()
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
    row = {
        "experiment": f"xgboost_{feature_version.lower()}",
        "family": "xgboost",
        "feature_version": feature_version,
        "trial": TRIAL_NUMBER,
        "fold": fold,
        "train_through": fold - 1,
        "trackman": False,
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
            "model": row["experiment"],
            "prediction": prediction.astype("float32"),
        }
    )
    del model, train_x, valid_x, train_y, valid_y, weights, prediction
    gc.collect()
    return row, pred


def main():
    frame, original, feature_sets = load_frame()
    trial = load_trial()
    results = []
    predictions = []

    # 2024 is the primary screen. Re-run 2023 only for variants that beat the
    # all-feature V2 result by a useful margin, plus the V1 reproduction anchor.
    screen = {}
    for name, additions in feature_sets.items():
        features = list(dict.fromkeys(original + additions))
        row, pred = run_one(frame, features, name, 2024, trial)
        results.append(row)
        predictions.append(pred)
        screen[name] = row["normalized_brier"]
        print(json.dumps(row, ensure_ascii=False), flush=True)

    v1_ratio = screen["V1_BASE_RECHECK"]
    candidates = ["V1_BASE_RECHECK"] + [
        name
        for name, _ in sorted(screen.items(), key=lambda item: item[1])
        if name != "V1_BASE_RECHECK" and screen[name] <= v1_ratio + 0.0015
    ][:5]
    for name in candidates:
        additions = feature_sets[name]
        features = list(dict.fromkeys(original + additions))
        row, pred = run_one(frame, features, name, 2023, trial)
        results.append(row)
        predictions.append(pred)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    pd.DataFrame(results).to_csv(WORK_DIR / "v2_ablation_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        WORK_DIR / "v2_ablation_predictions.parquet", index=False
    )
    summary = {
        "study": STUDY_NAME,
        "trial": TRIAL_NUMBER,
        "screen_fold": 2024,
        "cross_fold_candidates": candidates,
        "feature_sets": {key: value for key, value in feature_sets.items()},
        "results": results,
    }
    (WORK_DIR / "v2_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
