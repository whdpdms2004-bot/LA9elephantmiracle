from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
DATA_DIR = APP_DIR / "data"
OUTPUT_DIR = APP_DIR / "output"


def add_v1_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["count_state"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    x["runner_out_state"] = x["base_state"].astype(str) + "_o" + x["outs_before"].astype(str)
    x["handedness_matchup"] = x["pitcher_hand"].astype(str) + "_" + x["batter_hand"].astype(str)
    x["score_abs"] = x["score_diff_pitcher_team"].abs()
    x["late_inning"] = (x["inning"] >= 7).astype("int8")
    x["high_leverage"] = (x["li"] >= 2.0).astype("int8")
    for column in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]:
        x[f"log1p_{column}"] = np.log1p(x[column].clip(lower=0))
    for window in [1, 3, 5]:
        x[f"pitcher_success_delta_prev{window}"] = (
            x[f"asof_pitcher_prev{window}_game_success_rate"] - x["asof_pitcher_success_rate"]
        )
        x[f"pitcher_middle_delta_prev{window}"] = (
            x[f"asof_pitcher_prev{window}_game_middle_rate"] - x["asof_pitcher_middle_rate"]
        )
    x["ball_strike_rate_sum_gap"] = (
        x["asof_pitcher_ball_rate"] + x["asof_pitcher_strike_rate"] - 1.0
    )
    return x


def logit(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def sigmoid(value):
    z = np.clip(np.asarray(value, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def calibrate(probability, mode, params):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    if mode == "none":
        return p
    if mode == "logit_shift":
        return sigmoid(logit(p) + float(params["offset"]))
    if mode == "platt":
        coefficient = float(params["coef"][0])
        intercept = float(params["intercept"][0])
        return sigmoid(coefficient * logit(p) + intercept)
    if mode == "beta":
        coefficients = np.asarray(params["coef"], dtype=float)
        intercept = float(params["intercept"][0])
        value = coefficients[0] * np.log(p) + coefficients[1] * np.log1p(-p) + intercept
        return sigmoid(value)
    if mode == "isotonic":
        x = np.asarray(params["x_thresholds"], dtype=float)
        y = np.asarray(params["y_thresholds"], dtype=float)
        return np.interp(p, x, y, left=y[0], right=y[-1])
    raise ValueError(f"Unknown calibration mode: {mode}")


def main():
    metadata_path = MODEL_DIR / "metadata.json"
    test_path = DATA_DIR / "test.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing test data: {test_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    test = pd.read_csv(test_path)
    features = metadata["feature_columns"]
    categorical = metadata["categorical_columns"]
    engineered = add_v1_features(test)
    missing = [column for column in features if column not in engineered.columns]
    if missing:
        raise ValueError(f"Test columns are missing: {missing}")

    families = {item["family"] for item in metadata["models"]}
    xgb_frame = None
    cat_frame = None
    cat_pool = None
    if "xgboost" in families:
        xgb_frame = engineered[features].copy()
        for column in categorical:
            mapping = metadata["category_mappings"][column]
            xgb_frame[column] = (
                xgb_frame[column].fillna("__MISSING__").astype(str)
                .map(mapping).fillna(-1).astype("int32")
            )
        for column in features:
            xgb_frame[column] = pd.to_numeric(xgb_frame[column], errors="coerce").astype("float32")
    if "catboost" in families:
        from catboost import Pool

        cat_frame = engineered[features].copy()
        for column in categorical:
            cat_frame[column] = cat_frame[column].fillna("__MISSING__").astype(str)
        cat_pool = Pool(cat_frame, cat_features=categorical)

    model_predictions = []
    weights = []
    for item in metadata["models"]:
        model_path = MODEL_DIR / item["filename"]
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model: {model_path}")
        if item["family"] == "xgboost":
            # Do not use XGBClassifier at inference.  Its load/predict path
            # depends on scikit-learn estimator tags, which changed in
            # scikit-learn 1.8 on the evaluation server.  The native Booster
            # reads the same model and is independent of sklearn internals.
            import xgboost as xgb

            model = xgb.Booster(params={"device": "cpu", "nthread": 6})
            model.load_model(str(model_path))
            matrix = xgb.DMatrix(xgb_frame, feature_names=features)
            prediction = model.predict(matrix)
        elif item["family"] == "catboost":
            from catboost import CatBoostClassifier

            model = CatBoostClassifier()
            model.load_model(str(model_path))
            prediction = model.predict_proba(cat_pool, thread_count=6)[:, 1]
        else:
            raise ValueError(item["family"])
        model_predictions.append(np.asarray(prediction, dtype=float))
        weights.append(float(item["weight"]))

    matrix = np.column_stack(model_predictions)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    if metadata["blend_space"] == "probability":
        raw_prediction = matrix @ weights
    elif metadata["blend_space"] == "logit":
        raw_prediction = sigmoid(logit(matrix) @ weights)
    else:
        raise ValueError(metadata["blend_space"])
    prediction = calibrate(
        raw_prediction, metadata["calibration"], metadata["calibrator_params"]
    )
    prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise RuntimeError("Invalid prediction output")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"row_id": test["row_id"].to_numpy(), "control_success": prediction}
    ).to_csv(OUTPUT_DIR / "submission.csv", index=False)


if __name__ == "__main__":
    main()
