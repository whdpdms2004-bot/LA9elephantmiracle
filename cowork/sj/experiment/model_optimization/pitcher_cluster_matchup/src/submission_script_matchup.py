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
    if version == "tabm_t0":
        # TabM applies its own frozen fold-fitted preprocessing from model.pt.
        return test.copy()
    if version == "v1":
        return add_v1_features(test)
    if version not in {
        "enhanced",
        "insight_adjusted",
        "insight_success",
        "f_insight_adjusted",
    }:
        raise ValueError(version)
    output = add_v2_row_features(test)
    lookup = pd.read_csv(MODEL_DIR / metadata["trackman_lookup_file"])
    output = output.merge(lookup, on="pitcher_id", how="left", validate="many_to_one")
    output["tm500_available"] = output["tm500_available"].fillna(0).astype("int8")
    output["tm500_unavailable"] = output["tm500_unavailable"].fillna(1).astype("int8")
    output = enrich_trackman(output, metadata["trackman_columns"])
    if version in {"insight_adjusted", "insight_success", "f_insight_adjusted"}:
        output = add_insight_success_features(output, metadata)
    return output


def encode_xgboost(frame, spec, metadata):
    features = metadata["feature_sets"][spec["feature_version"]]
    output = frame[features].copy()
    mappings = metadata["category_mappings"][spec["feature_version"]]
    for column in metadata["categorical_columns"]:
        if column not in output:
            continue
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


def matchup_correction(test, metadata, metadata_key="matchup_correction"):
    """Return a frozen 2025 pitcher-type x batter-type residual correction."""
    spec = metadata_key if isinstance(metadata_key, dict) else metadata[metadata_key]
    pitcher = pd.read_csv(
        MODEL_DIR / spec["pitcher_lookup_file"],
        usecols=["pitcher_id", "pitcher_type"],
    )
    batter = pd.read_csv(
        MODEL_DIR / spec["batter_lookup_file"],
        usecols=["batter_id", "batter_hand", "batter_type"],
    )
    pair = pd.read_csv(MODEL_DIR / spec["pair_table_file"])
    ridge = json.loads(
        (MODEL_DIR / spec["ridge_file"]).read_text(encoding="utf-8")
    )

    output = test[
        ["row_id", "pitcher_id", "pitcher_hand", "batter_id", "batter_hand"]
    ].copy()
    output["__order"] = np.arange(len(output))
    output = output.merge(
        pitcher, on="pitcher_id", how="left", validate="many_to_one"
    )
    output["pitcher_type"] = output["pitcher_type"].fillna(
        "H" + output["pitcher_hand"].astype(str) + "_new"
    )
    output = output.merge(
        batter,
        on=["batter_id", "batter_hand"],
        how="left",
        validate="many_to_one",
    )
    output["batter_type"] = output["batter_type"].fillna(
        "BH" + output["batter_hand"].astype(str) + "_new"
    )
    output = output.merge(
        pair,
        on=["pitcher_type", "batter_type"],
        how="left",
        validate="many_to_one",
    ).sort_values("__order")
    feature_order = ridge["feature_order"]
    delta_column = feature_order[0]
    reliability_column = feature_order[1]
    known_column = feature_order[-1]
    output[known_column] = output[delta_column].notna().astype("float32")
    output[delta_column] = output[delta_column].fillna(0.0)
    output[reliability_column] = output[reliability_column].fillna(0.0)
    matrix = output[feature_order].to_numpy("float64")
    median = np.asarray(ridge["imputer_statistics"], dtype="float64")
    matrix = np.where(np.isfinite(matrix), matrix, median)
    mean = np.asarray(ridge["scaler_mean"], dtype="float64")
    scale = np.asarray(ridge["scaler_scale"], dtype="float64")
    coefficient = np.asarray(ridge["ridge_coef"], dtype="float64")
    correction = ((matrix - mean) / scale) @ coefficient
    low, high = ridge["correction_clip"]
    return np.clip(correction, low, high)


def r_context_correction(test, metadata):
    """Apply a frozen prior-OOF R-game context residual lookup."""
    spec = metadata["r_context_correction"]
    lookup_path = MODEL_DIR / spec["lookup_file"]
    if not lookup_path.is_file():
        raise FileNotFoundError(f"Missing R-context lookup: {lookup_path}")
    keys = list(spec["keys"])
    correction_column = spec.get("correction_column", "scaled_correction")
    lookup = pd.read_csv(lookup_path, usecols=keys + [correction_column])
    output = test[["row_id", "game_type", "inning", *[
        key for key in keys if key != "inning_bucket"
    ]]].copy()
    output["__order"] = np.arange(len(output))
    if "inning_bucket" in keys:
        output["inning_bucket"] = pd.cut(
            output["inning"], [0, 3, 6, 9, np.inf],
            labels=["1-3", "4-6", "7-9", "10+"], right=True,
        ).astype(str)
    output = output.merge(
        lookup, on=keys, how="left", validate="many_to_one"
    ).sort_values("__order")
    if not output["row_id"].reset_index(drop=True).equals(
        test["row_id"].reset_index(drop=True)
    ):
        raise RuntimeError("R-context lookup changed test row order")
    is_r = output["game_type"].eq(spec.get("game_type", "R")).to_numpy()
    correction = np.zeros(len(output), dtype="float64")
    correction[is_r] = output.loc[
        is_r, correction_column
    ].fillna(0.0).to_numpy(float)
    return correction


def game_type_expert_correction(test, metadata, logical_names, logical_predictions):
    """Return a partial-pooled local-expert residual for one game type."""
    spec = metadata["game_type_expert"]
    game_type = str(spec["game_type"])
    mask = test["game_type"].astype(str).eq(game_type).to_numpy()
    correction = np.zeros(len(test), dtype="float64")
    if not mask.any():
        return correction
    reference_name = spec["reference_model"]
    reference = np.asarray(
        logical_predictions[logical_names.index(reference_name)], dtype="float64"
    )
    for item in spec["experts"]:
        local = np.asarray(
            logical_predictions[logical_names.index(item["model_name"])],
            dtype="float64",
        )
        if not np.isfinite(local[mask]).all():
            raise RuntimeError(f"Invalid {game_type} expert prediction: {item['model_name']}")
        correction[mask] += float(item["weight"]) * (
            local[mask] - reference[mask]
        )
    return correction


def transform_tabm(frame, preprocessor):
    """Apply a serialized T0 FoldPreprocessor without importing training code."""
    cat_parts = []
    for column in preprocessor["cat_columns"]:
        mapping = preprocessor["cat_maps"][column]
        values = frame[column].astype("string").fillna("<NA>")
        cat_parts.append(
            values.map(mapping).fillna(0).to_numpy(dtype="int64", copy=False)
        )
    x_cat = (
        np.column_stack(cat_parts).astype("int64", copy=False)
        if cat_parts
        else np.empty((len(frame), 0), dtype="int64")
    )
    num_parts = []
    for transform in preprocessor["num_transforms"]:
        raw = pd.to_numeric(frame[transform["name"]], errors="coerce").to_numpy(
            dtype="float64"
        )
        raw[~np.isfinite(raw)] = np.nan
        if transform["log1p"]:
            raw = np.log1p(np.clip(raw, 0.0, None))
        missing = np.isnan(raw)
        filled = np.where(missing, float(transform["median"]), raw)
        scaled = (
            np.clip(
                filled,
                float(transform["lower"]),
                float(transform["upper"]),
            )
            - float(transform["mean"])
        ) / float(transform["std"])
        num_parts.append(scaled.astype("float32", copy=False))
        if transform["add_missing"]:
            num_parts.append(missing.astype("float32", copy=False))
    x_num = (
        np.column_stack(num_parts).astype("float32", copy=False)
        if num_parts
        else np.empty((len(frame), 0), dtype="float32")
    )
    return x_num, x_cat


def predict_tabm(frame, path, batch_size=4096):
    import tabm
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"Missing model: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = tabm.TabM.make(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    x_num, x_cat = transform_tabm(frame, checkpoint["preprocessor"])
    x_num = torch.from_numpy(np.ascontiguousarray(x_num))
    x_cat = torch.from_numpy(np.ascontiguousarray(x_cat))
    output = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            stop = min(len(frame), start + batch_size)
            xb_num = x_num[start:stop].to(device, non_blocking=True)
            xb_cat = x_cat[start:stop].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    xb_num if xb_num.shape[1] else None,
                    xb_cat if xb_cat.shape[1] else None,
                ).squeeze(-1)
            output.append(logits.float().sigmoid().mean(dim=1).cpu().numpy())
    return np.concatenate(output)


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
        route = item.get("route_game_type")
        route_mask = (
            test["game_type"].astype(str).eq(str(route)).to_numpy()
            if route is not None
            else np.ones(len(test), dtype=bool)
        )
        cache_key = (version, route)
        seed_predictions = []
        if route is not None and not route_mask.any():
            logical_predictions.append(
                np.full(len(test), np.nan, dtype="float64")
            )
            logical_weights.append(float(item["weight"]))
            logical_names.append(item["model_name"])
            continue
        if item["family"] == "xgboost":
            import xgboost as xgb

            if cache_key not in numeric_frames:
                numeric_frames[cache_key] = encode_xgboost(
                    frames[version].loc[route_mask], item, metadata
                )
            if cache_key not in xgb_matrices:
                xgb_matrices[cache_key] = xgb.DMatrix(
                    numeric_frames[cache_key], feature_names=features
                )
            for filename in item["filenames"]:
                path = MODEL_DIR / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing model: {path}")
                booster = xgb.Booster(params={"device": "cuda", "nthread": 6})
                booster.load_model(str(path))
                seed_predictions.append(booster.predict(xgb_matrices[cache_key]))
        elif item["family"] == "lightgbm":
            import lightgbm as lgb

            if cache_key not in numeric_frames:
                numeric_frames[cache_key] = encode_xgboost(
                    frames[version].loc[route_mask], item, metadata
                )
            for filename in item["filenames"]:
                path = MODEL_DIR / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing model: {path}")
                booster = lgb.Booster(model_file=str(path))
                seed_predictions.append(
                    booster.predict(numeric_frames[cache_key], num_threads=6)
                )
        elif item["family"] == "catboost":
            from catboost import CatBoostClassifier, Pool

            if cache_key not in cat_pools:
                cat_frame = frames[version].loc[route_mask, features].copy()
                categorical = [
                    column
                    for column in metadata["categorical_columns"]
                    if column in cat_frame
                ]
                for column in categorical:
                    cat_frame[column] = cat_frame[column].fillna("__MISSING__").astype(str)
                cat_pools[cache_key] = Pool(
                    cat_frame, cat_features=categorical
                )
            for filename in item["filenames"]:
                path = MODEL_DIR / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing model: {path}")
                model = CatBoostClassifier()
                model.load_model(str(path))
                seed_predictions.append(
                    model.predict_proba(cat_pools[cache_key], thread_count=6)[:, 1]
                )
        elif item["family"] == "tabm":
            for filename in item["filenames"]:
                seed_predictions.append(
                    predict_tabm(
                        frames[version].loc[route_mask],
                        MODEL_DIR / filename,
                        batch_size=int(item.get("batch_size", 4096)),
                    )
                )
        else:
            raise ValueError(item["family"])
        routed_prediction = np.mean(seed_predictions, axis=0)
        if route is not None:
            expanded = np.full(len(test), np.nan, dtype="float64")
            expanded[route_mask] = routed_prediction
            routed_prediction = expanded
        logical_predictions.append(routed_prediction)
        logical_weights.append(float(item["weight"]))
        logical_names.append(item["model_name"])

    outer = metadata.get("outer_blend")
    multi = metadata.get("multi_insight_blend")
    insight_name = outer.get("insight_model") if outer else None
    insight_names = {insight_name} if insight_name else set()
    if outer and outer.get("insight_anchor_model"):
        insight_names.add(outer["insight_anchor_model"])
    if outer and outer.get("insight_large_model"):
        insight_names.add(outer["insight_large_model"])
    if multi:
        insight_names.update(
            {multi["insight_anchor_model"], multi["insight_large_model"]}
        )
    expert = metadata.get("game_type_expert")
    if expert:
        insight_names.update(item["model_name"] for item in expert["experts"])
    base_indices = [
        index for index, name in enumerate(logical_names) if name not in insight_names
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
    if multi:
        anchor_index = logical_names.index(multi["insight_anchor_model"])
        large_index = logical_names.index(multi["insight_large_model"])
        anchor_prediction = np.asarray(
            logical_predictions[anchor_index], dtype=float
        )
        large_prediction = np.asarray(
            logical_predictions[large_index], dtype=float
        )

        current_prediction = anchor_prediction.copy()
        current_success = matchup_correction(
            test, metadata, metadata_key=multi["current_success_correction"]
        )
        current_reverse = np.mean(
            [
                matchup_correction(test, metadata, metadata_key=spec)
                for spec in multi["current_reverse_corrections"]
            ],
            axis=0,
        )
        current_prediction = np.clip(
            current_prediction
            + float(multi["current_success_scale"]) * current_success
            + float(multi["current_reverse_scale"]) * current_reverse,
            1e-6,
            1.0 - 1e-6,
        )

        large_prediction = (
            anchor_prediction
            + float(multi["large_base_weight"])
            * (large_prediction - anchor_prediction)
        )
        large_success = matchup_correction(
            test, metadata, metadata_key=multi["large_success_correction"]
        )
        large_reverse = np.mean(
            [
                matchup_correction(test, metadata, metadata_key=spec)
                for spec in multi["large_reverse_corrections"]
            ],
            axis=0,
        )
        large_prediction = np.clip(
            large_prediction
            + float(multi["large_success_scale"]) * large_success
            + float(multi["large_reverse_scale"]) * large_reverse,
            1e-6,
            1.0 - 1e-6,
        )
        blend_weights = np.asarray(
            [
                multi["base_weight"],
                multi["current_weight"],
                multi["large_weight"],
            ],
            dtype=float,
        )
        blend_weights /= blend_weights.sum()
        prediction = np.clip(
            np.column_stack(
                [base_prediction, current_prediction, large_prediction]
            )
            @ blend_weights,
            1e-6,
            1.0 - 1e-6,
        )
    elif outer:
        if outer.get("distribution_match"):
            anchor_index = logical_names.index(outer["insight_anchor_model"])
            large_index = logical_names.index(outer["insight_large_model"])
            anchor_prediction = np.asarray(
                logical_predictions[anchor_index], dtype=float
            )
            large_prediction = np.asarray(
                logical_predictions[large_index], dtype=float
            )
            mode = outer["distribution_match"]
            if mode == "probability_mean_std":
                source_std = max(float(large_prediction.std()), 1e-8)
                insight_prediction = (
                    (large_prediction - float(large_prediction.mean()))
                    * (float(anchor_prediction.std()) / source_std)
                    + float(anchor_prediction.mean())
                )
                insight_prediction = np.clip(
                    insight_prediction, 1e-6, 1.0 - 1e-6
                )
            else:
                raise ValueError(f"Unknown distribution match: {mode}")
        else:
            insight_index = logical_names.index(insight_name)
            insight_prediction = np.asarray(
                logical_predictions[insight_index], dtype=float
            )
        if metadata.get("matchup_correction"):
            correction = matchup_correction(test, metadata)
            correction_scale = float(
                metadata["matchup_correction"].get("correction_scale", 1.0)
            )
            insight_prediction = np.clip(
                insight_prediction + correction_scale * correction,
                1e-6,
                1.0 - 1e-6,
            )
        if metadata.get("reverse_matchup_correction"):
            reverse_correction = matchup_correction(
                test, metadata, metadata_key="reverse_matchup_correction"
            )
            reverse_scale = float(
                metadata["reverse_matchup_correction"].get("correction_scale", 1.0)
            )
            insight_prediction = np.clip(
                insight_prediction + reverse_scale * reverse_correction,
                1e-6,
                1.0 - 1e-6,
            )
        if metadata.get("reverse_matchup_corrections"):
            reverse_specs = metadata["reverse_matchup_corrections"]
            reverse_values = [
                matchup_correction(test, metadata, metadata_key=spec)
                for spec in reverse_specs
            ]
            reverse_average = np.mean(reverse_values, axis=0)
            reverse_scale = float(metadata.get("reverse_matchup_scale", 1.0))
            insight_prediction = np.clip(
                insight_prediction + reverse_scale * reverse_average,
                1e-6,
                1.0 - 1e-6,
            )
        insight_weight = float(outer["insight_weight"])
        prediction = np.clip(
            insight_weight * insight_prediction
            + (1.0 - insight_weight) * base_prediction,
            1e-6,
            1.0 - 1e-6,
        )
    else:
        prediction = base_prediction
    if metadata.get("r_context_correction"):
        prediction = np.clip(
            prediction + r_context_correction(test, metadata),
            1e-6,
            1.0 - 1e-6,
        )
    if metadata.get("game_type_expert"):
        prediction = np.clip(
            prediction
            + game_type_expert_correction(
                test, metadata, logical_names, logical_predictions
            ),
            1e-6,
            1.0 - 1e-6,
        )
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise RuntimeError("Invalid prediction output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"row_id": test["row_id"].to_numpy(), "control_success": prediction}
    ).to_csv(OUTPUT_DIR / "submission.csv", index=False)


if __name__ == "__main__":
    main()
