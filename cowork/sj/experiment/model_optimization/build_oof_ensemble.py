from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    ROOT,
    SEED,
    TARGET,
    encode_xgboost_fold,
    load_frame,
    prepare_catboost_frame,
    probability_metrics,
    recency_weights,
)


WORK_DIR = ROOT / "experiment" / "model_optimization"
FOLDS = [2022, 2023, 2024]
TRANSITIONS = [(2022, 2023), (2023, 2024)]
TOP_ROBUST_PER_FAMILY = 4
TOP_RECENT_PER_FAMILY = 4
STUDIES = {
    "xgboost": "xgboost_v1_full_2023_2024",
    "catboost": "catboost_v1_full_2023_2024",
}


def load_top_trials(family, study_name):
    storage = f"sqlite:///{(WORK_DIR / f'{study_name}.db').as_posix()}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(f"No completed trials: {study_name}")

    # The robust objective protects against a lucky recent-season result, while
    # fold_2024 is the closest proxy for the hidden future season.  Keep the
    # union so the ensemble sees both stable and high-ceiling candidates.
    robust = sorted(complete, key=lambda t: t.value)[:TOP_ROBUST_PER_FAMILY]
    recent = sorted(
        complete,
        key=lambda t: t.user_attrs.get("fold_2024", {}).get("normalized_brier", np.inf),
    )[:TOP_RECENT_PER_FAMILY]
    selected = []
    seen = set()
    for trial in robust + recent:
        if trial.number not in seen:
            selected.append(trial)
            seen.add(trial.number)
    return selected


def train_xgb_trial(trial, encoded_folds):
    params = dict(trial.params)
    half_life = float(params.pop("half_life"))
    outputs = {}
    best_iterations = {}
    for fold_index, year in enumerate(FOLDS):
        data = encoded_folds[year]
        weights = recency_weights(data["train_season"], year, half_life)
        model = XGBClassifier(
            **params,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            device="cuda", random_state=SEED + trial.number + fold_index,
            n_jobs=6, early_stopping_rounds=220,
        )
        model.fit(
            data["train_x"], data["train_y"], sample_weight=weights,
            eval_set=[(data["valid_x"], data["valid_y"])], verbose=False,
        )
        outputs[year] = model.predict_proba(data["valid_x"])[:, 1].astype("float32")
        best_iterations[year] = int(model.best_iteration)
        del model
        gc.collect()
    return outputs, best_iterations


def train_cat_trial(trial, frame, cat_frame):
    params = dict(trial.params)
    half_life = float(params.pop("half_life"))
    outputs = {}
    best_iterations = {}
    for fold_index, year in enumerate(FOLDS):
        train_mask = frame["season"].lt(year)
        valid_mask = frame["season"].eq(year)
        weights = recency_weights(frame.loc[train_mask, "season"], year, half_life)
        train_pool = Pool(
            cat_frame.loc[train_mask], label=frame.loc[train_mask, TARGET],
            cat_features=CATEGORICAL_COLUMNS, weight=weights,
        )
        valid_pool = Pool(
            cat_frame.loc[valid_mask], label=frame.loc[valid_mask, TARGET],
            cat_features=CATEGORICAL_COLUMNS,
        )
        model = CatBoostClassifier(
            **params,
            loss_function="Logloss", eval_metric="Logloss", task_type="GPU", devices="0",
            random_seed=SEED + trial.number + fold_index,
            verbose=False, allow_writing_files=False,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=220)
        outputs[year] = model.predict_proba(valid_pool)[:, 1].astype("float32")
        best_iterations[year] = int(model.get_best_iteration())
        del model, train_pool, valid_pool
        gc.collect()
    return outputs, best_iterations


def logit(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def sigmoid(value):
    z = np.clip(np.asarray(value, dtype=float), -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def blend(matrix, weights, space):
    if space == "probability":
        return np.asarray(matrix) @ weights
    if space == "logit":
        return sigmoid(logit(matrix) @ weights)
    raise ValueError(space)


def optimize_weights(matrix, target, space, l2_strength=0.0):
    n_models = matrix.shape[1]
    initial = np.repeat(1.0 / n_models, n_models)

    def loss(weights):
        prediction = blend(matrix, weights, space)
        brier = float(np.mean((prediction - target) ** 2))
        penalty = float(l2_strength * np.sum((weights - initial) ** 2))
        return brier + penalty

    result = minimize(
        loss, initial, method="SLSQP", bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


def fit_calibrator(mode, target, prediction):
    y = np.asarray(target)
    p = np.clip(np.asarray(prediction), 1e-6, 1 - 1e-6)
    if mode == "none":
        return lambda values: np.clip(np.asarray(values), 1e-6, 1 - 1e-6), {}
    if mode == "logit_shift":
        result = minimize_scalar(
            lambda offset: np.mean((sigmoid(logit(p) + offset) - y) ** 2),
            bounds=(-2.0, 2.0), method="bounded",
        )
        offset = float(result.x)
        return lambda values: sigmoid(logit(values) + offset), {"offset": offset}
    if mode == "platt":
        model = LogisticRegression(C=1e6, solver="lbfgs").fit(logit(p).reshape(-1, 1), y)
        params = {"coef": model.coef_.ravel().tolist(), "intercept": model.intercept_.tolist()}
        return lambda values: model.predict_proba(logit(values).reshape(-1, 1))[:, 1], params
    if mode == "beta":
        design = np.column_stack([np.log(p), np.log1p(-p)])
        model = LogisticRegression(C=1e6, solver="lbfgs").fit(design, y)
        params = {"coef": model.coef_.ravel().tolist(), "intercept": model.intercept_.tolist()}
        return lambda values: model.predict_proba(
            np.column_stack([
                np.log(np.clip(values, 1e-6, 1 - 1e-6)),
                np.log1p(-np.clip(values, 1e-6, 1 - 1e-6)),
            ])
        )[:, 1], params
    if mode == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        params = {
            "x_thresholds": model.X_thresholds_.tolist(),
            "y_thresholds": model.y_thresholds_.tolist(),
        }
        return lambda values: model.predict(np.asarray(values)), params
    raise ValueError(mode)


def main():
    started = time.time()
    frame, features = load_frame(0)
    xgb_folds = {year: encode_xgboost_fold(frame, features, year) for year in FOLDS}
    cat_frame = prepare_catboost_frame(frame, features)
    predictions = {year: {} for year in FOLDS}
    selected = []

    for family, study_name in STUDIES.items():
        trials = load_top_trials(family, study_name)
        for rank, trial in enumerate(trials, start=1):
            model_name = f"{family}_trial_{trial.number}_rank_{rank}"
            print(f"training {model_name}", flush=True)
            if family == "xgboost":
                output, iterations = train_xgb_trial(trial, xgb_folds)
            else:
                output, iterations = train_cat_trial(trial, frame, cat_frame)
            for year in FOLDS:
                predictions[year][model_name] = output[year]
            selected.append({
                "model_name": model_name,
                "family": family,
                "trial": trial.number,
                "study": study_name,
                "objective": trial.value,
                "params": trial.params,
                "best_iterations": iterations,
            })

    model_columns = [item["model_name"] for item in selected]
    oof_parts = []
    individual_rows = []
    for year in FOLDS:
        mask = frame["season"].eq(year)
        part = frame.loc[mask, ["row_id", "season", TARGET]].reset_index(drop=True)
        for name in model_columns:
            part[name] = predictions[year][name]
            metric = probability_metrics(part[TARGET], part[name])
            individual_rows.append({"season": year, "model_name": name, **metric})
        oof_parts.append(part)
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_parquet(WORK_DIR / "top_models_oof_2023_2024.parquet", index=False)
    pd.DataFrame(individual_rows).to_csv(WORK_DIR / "top_models_individual_metrics.csv", index=False)

    candidates = []
    for space in ["probability", "logit"]:
        for weight_method, l2_strength in [
            ("equal", None),
            ("optimized", 0.0),
            ("optimized_l2_1e-4", 1e-4),
            ("optimized_l2_1e-3", 1e-3),
            ("optimized_l2_1e-2", 1e-2),
        ]:
            transition_inputs = []
            for calibration_year, validation_year in TRANSITIONS:
                calibration = oof[oof["season"].eq(calibration_year)]
                validation = oof[oof["season"].eq(validation_year)]
                cal_x = calibration[model_columns].to_numpy(float)
                val_x = validation[model_columns].to_numpy(float)
                cal_y = calibration[TARGET].to_numpy("int8")
                val_y = validation[TARGET].to_numpy("int8")
                if weight_method == "equal":
                    weights = np.repeat(1.0 / len(model_columns), len(model_columns))
                else:
                    weights = optimize_weights(cal_x, cal_y, space, l2_strength)
                transition_inputs.append({
                    "calibration_year": calibration_year,
                    "validation_year": validation_year,
                    "cal_y": cal_y,
                    "val_y": val_y,
                    "weights": weights,
                    "raw_cal": blend(cal_x, weights, space),
                    "raw_val": blend(val_x, weights, space),
                })
            for mode in ["none", "logit_shift", "platt", "beta", "isotonic"]:
                transition_results = {}
                ratios = []
                for item in transition_inputs:
                    calibrator, calibrator_params = fit_calibrator(
                        mode, item["cal_y"], item["raw_cal"]
                    )
                    metrics = probability_metrics(item["val_y"], calibrator(item["raw_val"]))
                    ratios.append(metrics["normalized_brier"])
                    key = f"{item['calibration_year']}_to_{item['validation_year']}"
                    transition_results[key] = {
                        "weights": item["weights"].tolist(),
                        "calibrator_params": calibrator_params,
                        **metrics,
                    }
                weighted = 0.35 * ratios[0] + 0.65 * ratios[1]
                objective = 0.80 * weighted + 0.20 * max(ratios)
                candidates.append({
                    "space": space,
                    "weight_method": weight_method,
                    "l2_strength": l2_strength,
                    "calibration": mode,
                    "objective": float(objective),
                    "bss_2023": transition_results["2022_to_2023"]["bss"],
                    "bss_2024": transition_results["2023_to_2024"]["bss"],
                    "normalized_brier_2023": ratios[0],
                    "normalized_brier_2024": ratios[1],
                    "transition_parameters": transition_results,
                })

    candidates = sorted(candidates, key=lambda row: row["objective"])
    best = candidates[0]

    # Refit the chosen meta-model on the latest available OOF season for 2025.
    deployment_frame = oof[oof["season"].eq(2024)]
    deployment_x = deployment_frame[model_columns].to_numpy(float)
    deployment_y = deployment_frame[TARGET].to_numpy("int8")
    if best["weight_method"] == "equal":
        deployment_weights = np.repeat(1.0 / len(model_columns), len(model_columns))
    else:
        deployment_weights = optimize_weights(
            deployment_x, deployment_y, best["space"], best["l2_strength"]
        )
    deployment_raw = blend(deployment_x, deployment_weights, best["space"])
    _, deployment_calibrator = fit_calibrator(
        best["calibration"], deployment_y, deployment_raw
    )
    deployment = {
        "space": best["space"],
        "weight_method": best["weight_method"],
        "l2_strength": best["l2_strength"],
        "calibration": best["calibration"],
        "weights": deployment_weights.tolist(),
        "calibrator_params": deployment_calibrator,
    }
    result = {
        "created_at_epoch": time.time(),
        "elapsed_sec": time.time() - started,
        "model_columns": model_columns,
        "selected_models": selected,
        "candidates": candidates,
        "best": best,
        "deployment": deployment,
    }
    (WORK_DIR / "ensemble_selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {k: v for k, v in row.items() if k not in {"weights", "calibrator_params"}}
        for row in candidates
    ]).to_csv(WORK_DIR / "ensemble_candidates.csv", index=False)
    print(json.dumps({"best": best, "deployment": deployment}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
