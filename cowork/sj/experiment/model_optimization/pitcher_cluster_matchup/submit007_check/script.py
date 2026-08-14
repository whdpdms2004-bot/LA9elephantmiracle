from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
DATA_DIR = APP_DIR / "data"
OUTPUT_DIR = APP_DIR / "output"


ROW_RATE_SPECS = {
    "pitcher_success": ("asof_pitcher_success_rate", "asof_pitcher_n", 0.50),
    "pitcher_reverse": ("asof_pitcher_reverse_rate", "asof_pitcher_n", 0.23),
    "pitcher_middle": ("asof_pitcher_middle_rate", "asof_pitcher_n", 0.15),
    "pitcher_ball": ("asof_pitcher_ball_rate", "asof_pitcher_n", 0.50),
    "pitcher_strike": ("asof_pitcher_strike_rate", "asof_pitcher_n", 0.50),
    "batter_success": ("asof_batter_success_rate", "asof_batter_n", 0.50),
    "batter_middle": ("asof_batter_middle_rate", "asof_batter_n", 0.15),
    "pitcher_fastball": ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n", 0.50),
    "pitcher_breaking": ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n", 0.30),
    "pitcher_offspeed": ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n", 0.20),
}


def add_v1_features(frame):
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


def add_v2_row_features(frame):
    output = add_v1_features(frame)
    for prefix, (rate_column, count_column, prior) in ROW_RATE_SPECS.items():
        rate = output[rate_column].astype(float)
        count = output[count_column].clip(lower=0).astype(float)
        output[f"{prefix}_is_missing"] = rate.isna().astype("int8")
        for strength in [10, 50, 200, 500]:
            output[f"{prefix}_smoothed_{strength}"] = (
                (rate.fillna(prior) * count + prior * strength) / (count + strength)
            ).astype("float32")
            output[f"{prefix}_reliability_{strength}"] = (
                count / (count + strength)
            ).astype("float32")
    pitcher_n = output["asof_pitcher_n"]
    batter_n = output["asof_batter_n"]
    for threshold in [0, 25, 100, 500, 1000]:
        operator = "eq" if threshold == 0 else "le"
        output[f"pitcher_n_{operator}_{threshold}"] = getattr(pitcher_n, operator)(
            threshold
        ).astype("int8")
        output[f"batter_n_{operator}_{threshold}"] = getattr(batter_n, operator)(
            threshold
        ).astype("int8")
    success_recent = output[
        [
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
        ]
    ]
    middle_recent = output[
        [
            "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate",
            "asof_pitcher_prev5_game_middle_rate",
        ]
    ]
    output["pitcher_recent_success_mean"] = success_recent.mean(axis=1).astype("float32")
    output["pitcher_recent_success_std"] = success_recent.std(axis=1).astype("float32")
    output["pitcher_recent_success_range"] = (
        success_recent.max(axis=1) - success_recent.min(axis=1)
    ).astype("float32")
    output["pitcher_recent_middle_mean"] = middle_recent.mean(axis=1).astype("float32")
    output["pitcher_recent_middle_std"] = middle_recent.std(axis=1).astype("float32")
    output["pitcher_recent_middle_range"] = (
        middle_recent.max(axis=1) - middle_recent.min(axis=1)
    ).astype("float32")
    output["pitcher_failure_rate_sum"] = (
        output["asof_pitcher_reverse_rate"]
        + output["asof_pitcher_middle_rate"]
        + 1.0
        - output["asof_pitcher_success_rate"]
    ).astype("float32")
    output["pitcher_control_component_gap"] = (
        output["asof_pitcher_success_rate"]
        + output["asof_pitcher_reverse_rate"]
        + output["asof_pitcher_middle_rate"]
        - 1.0
    ).astype("float32")
    return output


def enrich_trackman(frame, tm_columns):
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
    quality = output["cw_match_seasons"].ge(2) | (
        output["cw_mean_sim"].ge(0.90) & output["cw_min_margin"].ge(0.10)
    )
    output["tm500_high_confidence"] = quality.fillna(False).astype("int8")
    output["tm500_low_confidence"] = (
        output["tm500_available"].eq(1) & ~quality.fillna(False)
    ).astype("int8")
    return output


def add_insight_success_features(frame, metadata):
    output = frame.copy()
    constants = metadata["insight_prior_constants"]
    for prefix in ["pitcher_success", "batter_success"]:
        rate_column, count_column, _ = ROW_RATE_SPECS[prefix]
        values = constants[prefix]
        prior_last = float(values["prior_last"])
        raw = output[rate_column].astype(float)
        count = output[count_column].clip(lower=0).astype(float)
        raw_fill = raw.fillna(prior_last)
        adjusted = sigmoid(logit(raw_fill) + float(values["gap_logit_last"]))
        output[f"{prefix}_adjusted_smoothed_200"] = (
            (adjusted * count + prior_last * 200.0) / (count + 200.0)
        ).astype("float32")
        if metadata.get("insight_feature_mode") != "success_full":
            continue
        for name, prior in [
            ("last", prior_last),
            ("ewm1", float(values["prior_ewm1"])),
            ("ewm2", float(values["prior_ewm2"])),
        ]:
            output[f"{prefix}_dynamic_{name}_smoothed_200"] = (
                (raw_fill * count + prior * 200.0) / (count + 200.0)
            ).astype("float32")
        for strength in [150.0, 250.0]:
            output[f"{prefix}_dynamic_last_smoothed_{int(strength)}"] = (
                (raw_fill * count + prior_last * strength) / (count + strength)
            ).astype("float32")
    return output


def build_feature_frame(test, version, metadata):
    if version == "v1":
        return add_v1_features(test)
    if version not in {"enhanced", "insight_adjusted", "insight_success"}:
        raise ValueError(version)
    output = add_v2_row_features(test)
    lookup = pd.read_csv(MODEL_DIR / metadata["trackman_lookup_file"])
    output = output.merge(lookup, on="pitcher_id", how="left", validate="many_to_one")
    output["tm500_available"] = output["tm500_available"].fillna(0).astype("int8")
    output["tm500_unavailable"] = output["tm500_unavailable"].fillna(1).astype("int8")
    output = enrich_trackman(output, metadata["trackman_columns"])
    if version in {"insight_adjusted", "insight_success"}:
        output = add_insight_success_features(output, metadata)
    return output


def encode_xgboost(frame, spec, metadata):
    features = metadata["feature_sets"][spec["feature_version"]]
    output = frame[features].copy()
    mappings = metadata["category_mappings"][spec["feature_version"]]
    for column in metadata["categorical_columns"]:
        output[column] = (
            output[column]
            .fillna("__MISSING__")
            .astype(str)
            .map(mappings[column])
            .fillna(-1)
            .astype("int32")
        )
    for column in features:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("float32")
    return output


def logit(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def calibration_groups(test, mode, metadata):
    if not mode.startswith("cohort_"):
        return None
    rookie = test["asof_pitcher_n"].fillna(0).le(500).to_numpy("int8")
    if mode == "cohort_rookie_logit":
        return rookie
    count = (
        test["balls_before"].to_numpy("int8") * 3
        + test["strikes_before"].to_numpy("int8")
    )
    if mode == "cohort_count_logit":
        return count
    if mode == "cohort_count_hand_logit":
        hand = test["pitcher_hand"].astype(str).eq("R").to_numpy("int8")
        return (2 * count + hand).astype("int8")
    lookup = pd.read_csv(MODEL_DIR / metadata["trackman_lookup_file"], usecols=["pitcher_id"])
    available_pitchers = set(lookup["pitcher_id"].tolist())
    tm = test["pitcher_id"].isin(available_pitchers).to_numpy("int8")
    if mode == "cohort_tm_logit":
        return tm
    if mode == "cohort_tm_rookie_logit":
        return (2 * tm + rookie).astype("int8")
    raise ValueError(mode)


def calibrate(probability, mode, params, groups=None):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    if mode == "none":
        return p
    if mode == "logit_shift":
        return sigmoid(logit(p) + float(params["offset"]))
    if mode == "platt":
        return sigmoid(float(params["coef"][0]) * logit(p) + float(params["intercept"][0]))
    if mode == "beta":
        coefficients = np.asarray(params["coef"], dtype=float)
        return sigmoid(
            coefficients[0] * np.log(p)
            + coefficients[1] * np.log1p(-p)
            + float(params["intercept"][0])
        )
    if mode == "isotonic":
        x = np.asarray(params["x_thresholds"], dtype=float)
        y = np.asarray(params["y_thresholds"], dtype=float)
        return np.interp(p, x, y, left=y[0], right=y[-1])
    if mode.startswith("cohort_"):
        global_offset = float(params["global_offset"])
        offset = np.full(len(p), global_offset, dtype=float)
        if groups is not None:
            for code, value in params["group_offsets"].items():
                offset[np.asarray(groups) == int(code)] = float(value)
        return sigmoid(logit(p) + offset)
    raise ValueError(mode)


def main():
    metadata_path = MODEL_DIR / "metadata.json"
    test_path = DATA_DIR / "test.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing test data: {test_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    test = pd.read_csv(test_path)
    versions = sorted({item["feature_version"] for item in metadata["models"]})
    frames = {version: build_feature_frame(test, version, metadata) for version in versions}
    numeric_frames = {}
    xgb_matrices = {}
    cat_pools = {}
    logical_predictions = []
    logical_weights = []
    logical_names = []

    for item in metadata["models"]:
        version = item["feature_version"]
        features = metadata["feature_sets"][version]
        seed_predictions = []
        if item["family"] == "xgboost":
            import xgboost as xgb

            if version not in numeric_frames:
                numeric_frames[version] = encode_xgboost(frames[version], item, metadata)
            if version not in xgb_matrices:
                xgb_matrices[version] = xgb.DMatrix(
                    numeric_frames[version], feature_names=features
                )
            for filename in item["filenames"]:
                path = MODEL_DIR / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing model: {path}")
                booster = xgb.Booster(params={"device": "cuda", "nthread": 6})
                booster.load_model(str(path))
                seed_predictions.append(booster.predict(xgb_matrices[version]))
        elif item["family"] == "lightgbm":
            import lightgbm as lgb

            if version not in numeric_frames:
                numeric_frames[version] = encode_xgboost(frames[version], item, metadata)
            for filename in item["filenames"]:
                path = MODEL_DIR / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing model: {path}")
                booster = lgb.Booster(model_file=str(path))
                seed_predictions.append(
                    booster.predict(numeric_frames[version], num_threads=6)
                )
        elif item["family"] == "catboost":
            from catboost import CatBoostClassifier, Pool

            if version not in cat_pools:
                cat_frame = frames[version][features].copy()
                for column in metadata["categorical_columns"]:
                    cat_frame[column] = cat_frame[column].fillna("__MISSING__").astype(str)
                cat_pools[version] = Pool(
                    cat_frame, cat_features=metadata["categorical_columns"]
                )
            for filename in item["filenames"]:
                path = MODEL_DIR / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing model: {path}")
                model = CatBoostClassifier()
                model.load_model(str(path))
                seed_predictions.append(
                    model.predict_proba(cat_pools[version], thread_count=6)[:, 1]
                )
        else:
            raise ValueError(item["family"])
        logical_predictions.append(np.mean(seed_predictions, axis=0))
        logical_weights.append(float(item["weight"]))
        logical_names.append(item["model_name"])

    outer = metadata.get("outer_blend")
    insight_name = outer.get("insight_model") if outer else None
    base_indices = [
        index for index, name in enumerate(logical_names) if name != insight_name
    ]
    matrix = np.column_stack([logical_predictions[index] for index in base_indices])
    weights = np.asarray([logical_weights[index] for index in base_indices], dtype=float)
    weights /= weights.sum()
    if metadata["blend_space"] == "probability":
        raw = matrix @ weights
    elif metadata["blend_space"] == "logit":
        raw = sigmoid(logit(matrix) @ weights)
    else:
        raise ValueError(metadata["blend_space"])
    base_prediction = np.clip(
        calibrate(
            raw,
            metadata["calibration"],
            metadata["calibrator_params"],
            groups=calibration_groups(test, metadata["calibration"], metadata),
        ),
        1e-6,
        1.0 - 1e-6,
    )
    if outer:
        insight_index = logical_names.index(insight_name)
        insight_prediction = np.asarray(logical_predictions[insight_index], dtype=float)
        insight_weight = float(outer["insight_weight"])
        prediction = np.clip(
            insight_weight * insight_prediction
            + (1.0 - insight_weight) * base_prediction,
            1e-6,
            1.0 - 1e-6,
        )
    else:
        prediction = base_prediction
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise RuntimeError("Invalid prediction output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"row_id": test["row_id"].to_numpy(), "control_success": prediction}
    ).to_csv(OUTPUT_DIR / "submission.csv", index=False)


if __name__ == "__main__":
    main()
