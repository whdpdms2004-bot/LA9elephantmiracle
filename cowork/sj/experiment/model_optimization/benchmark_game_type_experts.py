from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier

from benchmark_insight_features import add_calibration_features, build_past_only_lookups
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import CATEGORICAL_COLUMNS, ROOT, SEED, TARGET, probability_metrics, recency_weights


WORK_DIR = ROOT / "experiment" / "model_optimization"
OUTPUT_DIR = WORK_DIR / "game_type_experts"
STUDY_NAME = "xgboost_v2r200_tm500_local_2024"
TRIAL_NUMBER = 93
FOLDS = (2023, 2024)


def load_frame_and_features():
    frame, base_features = load_enhanced_frame()
    labels = pd.read_parquet(WORK_DIR / "failure_component_labels.parquet")
    lookups, audit = build_past_only_lookups(frame, labels)
    if not all(
        item["source_season"] is None or item["source_season"] < item["target_season"]
        for item in audit
    ):
        raise RuntimeError("Past-only insight feature audit failed")
    frame, _, prior_columns = add_calibration_features(frame, lookups)
    success_adjusted = [
        column
        for column in prior_columns
        if column.startswith(("pitcher_success_", "batter_success_"))
        and "_adjusted_smoothed_" in column
    ]
    features = list(dict.fromkeys(base_features + success_adjusted))
    local_features = [column for column in features if column != "game_type"]
    return frame, features, local_features


def load_anchor_params():
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{(WORK_DIR / f'{STUDY_NAME}.db').as_posix()}",
    )
    trial = next(item for item in study.trials if item.number == TRIAL_NUMBER)
    params = dict(trial.params)
    half_life = float(params.pop("half_life"))
    params["grow_policy"] = "lossguide"
    return params, half_life


def build_candidates(anchor, anchor_half_life):
    def candidate(name, group, half_life, **updates):
        params = dict(anchor)
        params.update(updates)
        return {
            "name": name,
            "group": group,
            "half_life": float(half_life),
            "params": params,
        }

    r_candidates = [
        candidate("r_anchor", "R", anchor_half_life),
        candidate("r_recent035", "R", 0.35),
        candidate("r_stable075", "R", 0.75),
        candidate(
            "r_diverse24",
            "R",
            anchor_half_life,
            n_estimators=9000,
            learning_rate=float(anchor["learning_rate"]) / 1.10,
            max_depth=8,
            max_leaves=24,
            subsample=0.90,
            colsample_bytree=0.70,
        ),
    ]
    f_candidates = [candidate("f_anchor", "F", anchor_half_life)]
    for half_life in (0.25, 0.50, 1.00):
        tag = str(half_life).replace(".", "p")
        f_candidates.append(
            candidate(
                f"f_compact6_h{tag}",
                "F",
                half_life,
                n_estimators=4000,
                learning_rate=0.006,
                max_depth=3,
                max_leaves=6,
                min_child_weight=20.0,
                subsample=0.92,
                colsample_bytree=0.80,
                colsample_bylevel=0.90,
                gamma=0.10,
                reg_alpha=0.50,
                reg_lambda=100.0,
                max_bin=256,
            )
        )
        f_candidates.append(
            candidate(
                f"f_compact10_h{tag}",
                "F",
                half_life,
                n_estimators=5000,
                learning_rate=0.005,
                max_depth=4,
                max_leaves=10,
                min_child_weight=40.0,
                subsample=0.90,
                colsample_bytree=0.75,
                colsample_bylevel=0.90,
                gamma=0.20,
                reg_alpha=1.0,
                reg_lambda=200.0,
                max_bin=256,
            )
        )
    return r_candidates, f_candidates


def encode_subset(frame, features, train_mask, valid_mask):
    train_x = frame.loc[train_mask, features].copy()
    valid_x = frame.loc[valid_mask, features].copy()
    for column in CATEGORICAL_COLUMNS:
        if column not in train_x:
            continue
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
    return train_x, valid_x


def fit_predict(frame, train_mask, valid_mask, train_x, valid_x, spec, fold):
    started = time.time()
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    weights = recency_weights(
        frame.loc[train_mask, "season"], fold, spec["half_life"]
    )
    model = XGBClassifier(
        **spec["params"],
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
    prediction = model.predict_proba(valid_x)[:, 1].astype("float32")
    result = {
        "model": spec["name"],
        "game_type": spec["group"],
        "fold": fold,
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "half_life": spec["half_life"],
        "best_iteration": int(model.best_iteration),
        "elapsed_sec": time.time() - started,
        **probability_metrics(valid_y, prediction),
    }
    del model, train_y, valid_y, weights
    gc.collect()
    return prediction, result


def metric_row(y, prediction, fold, name, group, **extra):
    return {
        "fold": fold,
        "candidate": name,
        "group": group,
        "n": len(y),
        **extra,
        **probability_metrics(y, prediction),
    }


def run_fold(frame, global_features, local_features, r_candidates, f_candidates, anchor, anchor_half_life, fold):
    valid_mask = frame["season"].eq(fold)
    train_mask = frame["season"].lt(fold)
    global_train_x, global_valid_x = encode_subset(
        frame, global_features, train_mask, valid_mask
    )
    global_spec = {
        "name": "global_anchor",
        "group": "ALL",
        "half_life": anchor_half_life,
        "params": anchor,
    }
    global_prediction, global_result = fit_predict(
        frame,
        train_mask,
        valid_mask,
        global_train_x,
        global_valid_x,
        global_spec,
        fold,
    )
    valid = frame.loc[valid_mask, ["row_id", "game_type", TARGET]].reset_index(drop=True)
    y = valid[TARGET].to_numpy("int8")
    is_r = valid["game_type"].astype(str).eq("R").to_numpy()
    metrics = [metric_row(y, global_prediction, fold, "global_anchor", "ALL")]
    metrics.append(metric_row(y[is_r], global_prediction[is_r], fold, "global_anchor", "R"))
    metrics.append(metric_row(y[~is_r], global_prediction[~is_r], fold, "global_anchor", "F"))
    model_results = [global_result]
    expert_predictions = []
    predictions = {"global_anchor": global_prediction}

    for group, candidates in (("R", r_candidates), ("F", f_candidates)):
        group_train = train_mask & frame["game_type"].astype(str).eq(group)
        group_valid = valid_mask & frame["game_type"].astype(str).eq(group)
        train_x, valid_x = encode_subset(frame, local_features, group_train, group_valid)
        position = is_r if group == "R" else ~is_r
        for spec in candidates:
            prediction, result = fit_predict(
                frame, group_train, group_valid, train_x, valid_x, spec, fold
            )
            predictions[spec["name"]] = prediction
            model_results.append(result)
            metrics.append(metric_row(y[position], prediction, fold, spec["name"], group))
            expert_predictions.append(
                pd.DataFrame(
                    {
                        "row_id": valid.loc[position, "row_id"].to_numpy(),
                        "season": fold,
                        "game_type": group,
                        TARGET: y[position],
                        "model": spec["name"],
                        "prediction": prediction,
                    }
                )
            )
        del train_x, valid_x
        gc.collect()

    alpha_grid = np.round(np.arange(0.0, 1.251, 0.25), 2)
    grid_rows = []
    for r_spec in r_candidates:
        r_prediction = predictions[r_spec["name"]]
        for f_spec in f_candidates:
            f_prediction = predictions[f_spec["name"]]
            separate = np.empty_like(global_prediction)
            separate[is_r] = r_prediction
            separate[~is_r] = f_prediction
            for alpha_r in alpha_grid:
                for alpha_f in alpha_grid:
                    gated = global_prediction.copy()
                    gated[is_r] += alpha_r * (separate[is_r] - global_prediction[is_r])
                    gated[~is_r] += alpha_f * (separate[~is_r] - global_prediction[~is_r])
                    gated = np.clip(gated, 1e-6, 1.0 - 1e-6)
                    name = f"{r_spec['name']}__{f_spec['name']}"
                    common = {
                        "r_model": r_spec["name"],
                        "f_model": f_spec["name"],
                        "alpha_r": alpha_r,
                        "alpha_f": alpha_f,
                    }
                    grid_rows.append(metric_row(y, gated, fold, name, "ALL", **common))
                    grid_rows.append(metric_row(y[is_r], gated[is_r], fold, name, "R", **common))
                    grid_rows.append(metric_row(y[~is_r], gated[~is_r], fold, name, "F", **common))

    global_frame = pd.DataFrame(
        {
            "row_id": valid["row_id"],
            "season": fold,
            "game_type": valid["game_type"].astype(str),
            TARGET: y,
            "model": "global_anchor",
            "prediction": global_prediction,
        }
    )
    del global_train_x, global_valid_x
    gc.collect()
    return metrics, model_results, grid_rows, [global_frame, *expert_predictions]


def robust_summary(grid):
    all_rows = grid.loc[grid["group"].eq("ALL")].copy()
    index = ["r_model", "f_model", "alpha_r", "alpha_f"]
    pivot = all_rows.pivot_table(index=index, columns="fold", values="normalized_brier")
    pivot = pivot.dropna(subset=list(FOLDS)).reset_index()
    pivot["robust_objective"] = (
        0.45 * pivot[2023]
        + 0.55 * pivot[2024]
        + 0.25 * (pivot[2023] - pivot[2024]).abs()
    )
    pivot["pure_trigger"] = pivot["alpha_r"].eq(1.0) & pivot["alpha_f"].eq(1.0)
    return pivot.sort_values("robust_objective").reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, global_features, local_features = load_frame_and_features()
    anchor, anchor_half_life = load_anchor_params()
    r_candidates, f_candidates = build_candidates(anchor, anchor_half_life)
    metrics = []
    model_results = []
    grid_rows = []
    predictions = []
    for fold in FOLDS:
        fold_metrics, fold_results, fold_grid, fold_predictions = run_fold(
            frame,
            global_features,
            local_features,
            r_candidates,
            f_candidates,
            anchor,
            anchor_half_life,
            fold,
        )
        metrics.extend(fold_metrics)
        model_results.extend(fold_results)
        grid_rows.extend(fold_grid)
        predictions.extend(fold_predictions)
        print(json.dumps({"fold": fold, "models_complete": len(fold_results)}, ensure_ascii=False), flush=True)

    metrics_frame = pd.DataFrame(metrics)
    model_frame = pd.DataFrame(model_results)
    grid_frame = pd.DataFrame(grid_rows)
    robust = robust_summary(grid_frame)
    metrics_frame.to_csv(OUTPUT_DIR / "expert_metrics.csv", index=False)
    model_frame.to_csv(OUTPUT_DIR / "training_metrics.csv", index=False)
    grid_frame.to_parquet(OUTPUT_DIR / "gating_grid.parquet", index=False)
    robust.to_csv(OUTPUT_DIR / "robust_gating_summary.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        OUTPUT_DIR / "expert_oof_predictions.parquet", index=False
    )
    pure = robust.loc[robust["pure_trigger"]]
    summary = {
        "feature_version": "INSIGHT_SUCCESS_ADJUSTED",
        "folds": list(FOLDS),
        "trackman_rule": "strictly before validation season; pitcher-season >=500",
        "global_feature_count": len(global_features),
        "local_feature_count": len(local_features),
        "r_candidates": [item["name"] for item in r_candidates],
        "f_candidates": [item["name"] for item in f_candidates],
        "best_robust": json.loads(robust.iloc[0].to_json()),
        "best_pure_trigger": json.loads(pure.iloc[0].to_json()),
        "outputs": {
            "metrics": "expert_metrics.csv",
            "training": "training_metrics.csv",
            "grid": "gating_grid.parquet",
            "robust": "robust_gating_summary.csv",
            "predictions": "expert_oof_predictions.parquet",
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
