from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    ROOT,
    SEED,
    TARGET,
    load_frame,
    prepare_catboost_frame,
    recency_weights,
)


WORK_DIR = ROOT / "experiment" / "model_optimization"
SELECTION_PATH = WORK_DIR / "ensemble_selection.json"
SUBMIT_DIR = WORK_DIR / "submit_optimized"
MODEL_DIR = SUBMIT_DIR / "model"


def encode_xgboost_full(frame, features):
    encoded = frame[features].copy()
    mappings = {}
    for column in CATEGORICAL_COLUMNS:
        values = encoded[column].fillna("__MISSING__").astype(str)
        mapping = {value: int(index) for index, value in enumerate(np.unique(values))}
        mappings[column] = mapping
        encoded[column] = values.map(mapping).astype("int32")
    for column in features:
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce").astype("float32")
    return encoded, mappings


def final_iterations(model_spec):
    best = model_spec["best_iterations"]
    recent = int(best.get("2024", best.get(2024, 0))) + 1
    return max(1, recent)


def main():
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    deployment = selection["deployment"]
    selected = selection["selected_models"]
    weights = np.asarray(deployment["weights"], dtype=float)
    if len(selected) != len(weights):
        raise RuntimeError("Model/weight length mismatch")

    # Truly zero-weight models cannot affect the ensemble and only waste inference time.
    active = [(spec, float(weight)) for spec, weight in zip(selected, weights) if weight > 1e-10]
    active_weights = np.asarray([weight for _, weight in active], dtype=float)
    active_weights /= active_weights.sum()

    if SUBMIT_DIR.exists():
        shutil.rmtree(SUBMIT_DIR)
    MODEL_DIR.mkdir(parents=True)

    frame, features = load_frame(0)
    target = frame[TARGET].to_numpy("int8")
    season = frame["season"].to_numpy("int16")
    families = {spec["family"] for spec, _ in active}
    xgb_frame = None
    category_mappings = {}
    cat_frame = None
    if "xgboost" in families:
        xgb_frame, category_mappings = encode_xgboost_full(frame, features)
    if "catboost" in families:
        cat_frame = prepare_catboost_frame(frame, features)

    model_metadata = []
    for model_index, ((spec, _), weight) in enumerate(zip(active, active_weights)):
        family = spec["family"]
        params = dict(spec["params"])
        half_life = float(params.pop("half_life"))
        iterations = final_iterations(spec)
        sample_weight = recency_weights(season, 2025, half_life)
        stem = f"{model_index:02d}_{family}_trial_{spec['trial']}"
        print(
            f"training {stem} iterations={iterations} weight={weight:.6f}",
            flush=True,
        )

        if family == "xgboost":
            params["n_estimators"] = iterations
            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                random_state=SEED + int(spec["trial"]),
                n_jobs=6,
            )
            model.fit(xgb_frame, target, sample_weight=sample_weight, verbose=False)
            filename = f"{stem}.ubj"
            model.save_model(str(MODEL_DIR / filename))
        elif family == "catboost":
            params["iterations"] = iterations
            # Pool construction is material, but weights differ by trial.
            train_pool = Pool(
                cat_frame,
                label=target,
                cat_features=CATEGORICAL_COLUMNS,
                weight=sample_weight,
            )
            model = CatBoostClassifier(
                **params,
                loss_function="Logloss",
                eval_metric="Logloss",
                task_type="GPU",
                devices="0",
                random_seed=SEED + int(spec["trial"]),
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(train_pool)
            filename = f"{stem}.cbm"
            model.save_model(str(MODEL_DIR / filename))
            del train_pool
        else:
            raise ValueError(family)

        model_metadata.append(
            {
                "model_name": spec["model_name"],
                "family": family,
                "trial": int(spec["trial"]),
                "filename": filename,
                "weight": float(weight),
                "half_life": half_life,
                "iterations": iterations,
            }
        )
        del model, sample_weight
        gc.collect()

    metadata = {
        "version": 2,
        "target": TARGET,
        "feature_columns": features,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "category_mappings": category_mappings,
        "models": model_metadata,
        "blend_space": deployment["space"],
        "calibration": deployment["calibration"],
        "calibrator_params": deployment["calibrator_params"],
        "selection_backtest": {
            "objective": selection["best"]["objective"],
            "bss_2023": selection["best"]["bss_2023"],
            "bss_2024": selection["best"]["bss_2024"],
        },
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata["selection_backtest"], indent=2), flush=True)


if __name__ == "__main__":
    main()
