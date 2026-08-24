from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from run_optuna_family import ROOT, SEED, TARGET, probability_metrics


WORK_DIR = ROOT / "experiment" / "model_optimization"
EMBED_DIR = ROOT / "experiment" / "pitcher_embedding" / "outputs" / "trackman500_multitask"
KEYS = ["row_id", "season"]
LONG_SOURCES = [
    ("enh", "enhanced_seed_oof_predictions.parquet"),
    ("v2", "v2_fixed_predictions.parquet"),
    ("tm", "trackman500_fixed_predictions.parquet"),
    ("catfix", "cat_enhanced_predictions.parquet"),
    ("pitchgrp", "trackman_pitchgroup_fixed_predictions.parquet"),
    ("gated", "trackman_gated_predictions.parquet"),
    ("smooth", "smoothing_grid_predictions.parquet"),
    ("direct", "direct_brier_predictions.parquet"),
]
# At most eight logical models keeps 245,789-row inference comfortably inside
# the ten-minute evaluation limit even when enhanced candidates use 3 seeds.
TOP_KS = [3, 5, 8]
CORRELATION_LIMITS = [0.999, 0.995, 0.990, 0.980]
WEIGHT_SPECS = [
    ("equal", None),
    ("optimized", 0.0),
    ("optimized_l2_1e-4", 1e-4),
    ("optimized_l2_1e-3", 1e-3),
    ("optimized_l2_1e-2", 1e-2),
]
SPACES = ["probability", "logit"]
CALIBRATIONS = [
    "none",
    "logit_shift",
    "platt",
    "beta",
    "isotonic",
    "cohort_tm_logit",
    "cohort_rookie_logit",
    "cohort_tm_rookie_logit",
    "cohort_count_logit",
    "cohort_count_hand_logit",
]


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def wide_from_long(path: Path, prefix: str):
    frame = pd.read_parquet(path)
    required = {*KEYS, "model", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    duplicates = frame.duplicated(KEYS + ["model"]).sum()
    if duplicates:
        raise ValueError(f"{path.name} has {duplicates} duplicate row/model keys")
    wide = frame.pivot(index=KEYS, columns="model", values="prediction").reset_index()
    wide.columns.name = None
    rename = {
        column: f"{prefix}__{safe_name(column)}"
        for column in wide.columns
        if column not in KEYS
    }
    return wide.rename(columns=rename), list(rename.values())


def load_pool():
    base_path = WORK_DIR / "top_models_oof_2023_2024.parquet"
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    base = pd.read_parquet(base_path)
    base_models = [column for column in base if column not in {*KEYS, TARGET}]
    base = base.rename(columns={column: f"v1__{safe_name(column)}" for column in base_models})
    model_columns = [f"v1__{safe_name(column)}" for column in base_models]
    inventory = [{"source": str(base_path.relative_to(ROOT)), "models": len(base_models)}]

    for prefix, filename in LONG_SOURCES:
        path = WORK_DIR / filename
        if not path.is_file():
            continue
        wide, columns = wide_from_long(path, prefix)
        base = base.merge(wide, on=KEYS, how="left", validate="one_to_one")
        model_columns.extend(columns)
        inventory.append({"source": str(path.relative_to(ROOT)), "models": len(columns)})

    if EMBED_DIR.is_dir():
        for path in sorted(EMBED_DIR.glob("multitask_dim*_oof_predictions.parquet")):
            match = re.search(r"dim(\d+)", path.name)
            dimension = match.group(1) if match else safe_name(path.stem)
            frame = pd.read_parquet(path)
            if frame.duplicated(KEYS).any():
                raise ValueError(f"duplicate embedding keys: {path}")
            column = f"embed__mt{dimension}"
            base = base.merge(
                frame[KEYS + ["prediction"]].rename(columns={"prediction": column}),
                on=KEYS,
                how="left",
                validate="one_to_one",
            )
            model_columns.append(column)
            inventory.append({"source": str(path.relative_to(ROOT)), "models": 1})

    model_columns = list(dict.fromkeys(model_columns))
    tm_path = WORK_DIR / "trackman500_asof_train.parquet"
    if tm_path.is_file():
        tm = pd.read_parquet(
            tm_path, columns=["row_id", "season", "tm500_available"]
        )
        base = base.merge(tm, on=KEYS, how="left", validate="one_to_one")
    else:
        base["tm500_available"] = 0
    raw = pd.read_csv(
        ROOT / "data" / "train.csv",
        usecols=[
            "row_id",
            "asof_pitcher_n",
            "balls_before",
            "strikes_before",
            "pitcher_hand",
        ],
    )
    base = base.merge(raw, on="row_id", how="left", validate="one_to_one")
    base["tm500_available"] = base["tm500_available"].fillna(0).astype("int8")
    if base.duplicated(KEYS).any():
        raise RuntimeError("OOF pool keys are not unique")
    return base, model_columns, inventory


def logit(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def sigmoid(value):
    z = np.clip(np.asarray(value, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def blend(matrix, weights, space):
    matrix = np.asarray(matrix, dtype=float)
    if space == "probability":
        return matrix @ weights
    if space == "logit":
        return sigmoid(logit(matrix) @ weights)
    raise ValueError(space)


def deterministic_sample(length, limit, salt):
    if length <= limit:
        return np.arange(length)
    rng = np.random.default_rng(SEED + salt)
    return np.sort(rng.choice(length, size=limit, replace=False))


def select_diverse_models(frame, candidates, top_k, correlation_limit, salt):
    y = frame[TARGET].to_numpy(float)
    losses = []
    for column in candidates:
        prediction = frame[column].to_numpy(float)
        if np.isfinite(prediction).all():
            losses.append((float(np.mean((prediction - y) ** 2)), column))
    ranked = [column for _, column in sorted(losses)]
    if not ranked:
        raise RuntimeError("No complete candidates for selection frame")
    sample = deterministic_sample(len(frame), 100_000, salt)
    sampled_y = y[sample]
    chosen = []
    residuals = []
    for column in ranked:
        residual = frame[column].to_numpy(float)[sample] - sampled_y
        if residuals:
            correlations = [
                abs(float(np.corrcoef(residual, existing)[0, 1]))
                for existing in residuals
            ]
            if max(correlations) > correlation_limit:
                continue
        chosen.append(column)
        residuals.append(residual)
        if len(chosen) >= min(top_k, len(ranked)):
            break
    if len(chosen) == 1 and len(ranked) > 1:
        chosen.append(ranked[1])
    return chosen


def optimize_weights(matrix, target, space, l2_strength, salt):
    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float)
    n_models = matrix.shape[1]
    initial = np.repeat(1.0 / n_models, n_models)
    sample = deterministic_sample(len(target), 150_000, salt)
    x = matrix[sample]
    y = target[sample]

    def loss(weights):
        prediction = blend(x, weights, space)
        brier = np.mean((prediction - y) ** 2)
        penalty = l2_strength * np.sum((weights - initial) ** 2)
        return float(brier + penalty)

    result = minimize(
        loss,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 1000, "ftol": 1e-13},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def cohort_values(frame, mode):
    if mode == "cohort_tm_logit":
        return frame["tm500_available"].to_numpy("int8")
    rookie = frame["asof_pitcher_n"].fillna(0).le(500).to_numpy("int8")
    if mode == "cohort_rookie_logit":
        return rookie
    if mode == "cohort_tm_rookie_logit":
        tm = frame["tm500_available"].to_numpy("int8")
        return (2 * tm + rookie).astype("int8")
    count = (
        frame["balls_before"].to_numpy("int8") * 3
        + frame["strikes_before"].to_numpy("int8")
    )
    if mode == "cohort_count_logit":
        return count
    if mode == "cohort_count_hand_logit":
        hand = frame["pitcher_hand"].astype(str).eq("R").to_numpy("int8")
        return (2 * count + hand).astype("int8")
    return None


def fit_calibrator(mode, target, prediction, salt, groups=None):
    y = np.asarray(target, dtype="int8")
    p = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1.0 - 1e-6)
    sample = deterministic_sample(len(y), 250_000, salt)
    ys = y[sample]
    ps = p[sample]
    if mode == "none":
        return {}, lambda values, values_group=None: np.clip(values, 1e-6, 1.0 - 1e-6)
    if mode == "logit_shift":
        result = minimize_scalar(
            lambda offset: np.mean((sigmoid(logit(ps) + offset) - ys) ** 2),
            bounds=(-1.0, 1.0),
            method="bounded",
        )
        offset = float(result.x)
        return {"offset": offset}, lambda values, values_group=None: sigmoid(logit(values) + offset)
    if mode == "platt":
        model = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000).fit(
            logit(ps).reshape(-1, 1), ys
        )
        params = {
            "coef": model.coef_.ravel().tolist(),
            "intercept": model.intercept_.ravel().tolist(),
        }
        return params, lambda values, values_group=None: model.predict_proba(
            logit(values).reshape(-1, 1)
        )[:, 1]
    if mode == "beta":
        design = np.column_stack([np.log(ps), np.log1p(-ps)])
        model = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000).fit(
            design, ys
        )
        params = {
            "coef": model.coef_.ravel().tolist(),
            "intercept": model.intercept_.ravel().tolist(),
        }
        return params, lambda values, values_group=None: model.predict_proba(
            np.column_stack(
                [
                    np.log(np.clip(values, 1e-6, 1.0 - 1e-6)),
                    np.log1p(-np.clip(values, 1e-6, 1.0 - 1e-6)),
                ]
            )
        )[:, 1]
    if mode == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(ps, ys)
        params = {
            "x_thresholds": model.X_thresholds_.tolist(),
            "y_thresholds": model.y_thresholds_.tolist(),
        }
        return params, lambda values, values_group=None: model.predict(np.asarray(values, dtype=float))
    if mode.startswith("cohort_"):
        if groups is None:
            raise ValueError(f"groups are required for {mode}")
        gs = np.asarray(groups)[sample]
        global_result = minimize_scalar(
            lambda offset: np.mean((sigmoid(logit(ps) + offset) - ys) ** 2),
            bounds=(-1.0, 1.0),
            method="bounded",
        )
        global_offset = float(global_result.x)
        codes = sorted(int(value) for value in np.unique(gs))
        code_to_index = {code: index for index, code in enumerate(codes)}
        group_index = np.asarray([code_to_index[int(value)] for value in gs], dtype="int8")

        def loss(offsets):
            applied = np.asarray(offsets)[group_index]
            brier = np.mean((sigmoid(logit(ps) + applied) - ys) ** 2)
            penalty = 1e-3 * np.mean((np.asarray(offsets) - global_offset) ** 2)
            return float(brier + penalty)

        result = minimize(
            loss,
            np.repeat(global_offset, len(codes)),
            method="L-BFGS-B",
            bounds=[(-1.0, 1.0)] * len(codes),
            options={"maxiter": 500, "ftol": 1e-14},
        )
        offsets = {str(code): float(result.x[index]) for index, code in enumerate(codes)}
        params = {
            "global_offset": global_offset,
            "group_offsets": offsets,
            "ridge": 1e-3,
        }

        def transform(values, values_group=None):
            if values_group is None:
                offset = np.repeat(global_offset, len(values))
            else:
                offset = np.full(len(values_group), global_offset, dtype=float)
                for code, value in offsets.items():
                    offset[np.asarray(values_group) == int(code)] = value
            return sigmoid(logit(values) + offset)

        return params, transform
    raise ValueError(mode)


def apply_calibrator(mode, params, prediction, groups=None):
    p = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1.0 - 1e-6)
    if mode == "none":
        return p
    if mode == "logit_shift":
        return sigmoid(logit(p) + float(params["offset"]))
    if mode == "platt":
        return sigmoid(
            float(params["coef"][0]) * logit(p) + float(params["intercept"][0])
        )
    if mode == "beta":
        coefficients = np.asarray(params["coef"], dtype=float)
        value = (
            coefficients[0] * np.log(p)
            + coefficients[1] * np.log1p(-p)
            + float(params["intercept"][0])
        )
        return sigmoid(value)
    if mode == "isotonic":
        x = np.asarray(params["x_thresholds"], dtype=float)
        y = np.asarray(params["y_thresholds"], dtype=float)
        return np.interp(p, x, y, left=y[0], right=y[-1])
    if mode.startswith("cohort_"):
        global_offset = float(params["global_offset"])
        offsets = params["group_offsets"]
        if groups is None:
            offset = np.repeat(global_offset, len(p))
        else:
            offset = np.full(len(groups), global_offset, dtype=float)
            for code, value in offsets.items():
                offset[np.asarray(groups) == int(code)] = value
        return sigmoid(logit(p) + offset)
    raise ValueError(mode)


def candidate_pool(frame, model_columns, years):
    subset = frame[frame["season"].isin(years)]
    return [column for column in model_columns if subset[column].notna().all()]


def evaluate_track(frame, pool, transitions, track_name):
    candidates = []
    setting_index = 0
    for space in SPACES:
        for weight_name, l2_strength in WEIGHT_SPECS:
            for top_k in TOP_KS:
                for correlation_limit in CORRELATION_LIMITS:
                    setting_index += 1
                    transition_cache = []
                    for calibration_year, validation_year in transitions:
                        calibration = frame[frame["season"].eq(calibration_year)]
                        validation = frame[frame["season"].eq(validation_year)]
                        selected = select_diverse_models(
                            calibration,
                            pool,
                            top_k,
                            correlation_limit,
                            salt=setting_index + calibration_year,
                        )
                        cal_x = calibration[selected].to_numpy(float)
                        val_x = validation[selected].to_numpy(float)
                        cal_y = calibration[TARGET].to_numpy("int8")
                        val_y = validation[TARGET].to_numpy("int8")
                        if weight_name == "equal":
                            weights = np.repeat(1.0 / len(selected), len(selected))
                        else:
                            weights = optimize_weights(
                                cal_x,
                                cal_y,
                                space,
                                float(l2_strength),
                                salt=setting_index + calibration_year * 10,
                            )
                        transition_cache.append(
                            {
                                "calibration_year": calibration_year,
                                "validation_year": validation_year,
                                "selected_models": selected,
                                "weights": weights,
                                "cal_y": cal_y,
                                "val_y": val_y,
                                "raw_cal": blend(cal_x, weights, space),
                                "raw_val": blend(val_x, weights, space),
                                "row_id": validation["row_id"].to_numpy(),
                                "calibration_frame": calibration,
                                "validation_frame": validation,
                            }
                        )
                    for calibration_mode in CALIBRATIONS:
                        transition_results = {}
                        ratios = []
                        for item in transition_cache:
                            params, calibrator = fit_calibrator(
                                calibration_mode,
                                item["cal_y"],
                                item["raw_cal"],
                                salt=setting_index + item["validation_year"] * 100,
                                groups=cohort_values(
                                    item["calibration_frame"], calibration_mode
                                ),
                            )
                            prediction = np.clip(
                                calibrator(
                                    item["raw_val"],
                                    cohort_values(
                                        item["validation_frame"], calibration_mode
                                    ),
                                ),
                                1e-6,
                                1.0 - 1e-6,
                            )
                            metrics = probability_metrics(item["val_y"], prediction)
                            ratios.append(metrics["normalized_brier"])
                            transition_results[
                                f"{item['calibration_year']}_to_{item['validation_year']}"
                            ] = {
                                "selected_models": item["selected_models"],
                                "weights": item["weights"].tolist(),
                                "calibrator_params": params,
                                **metrics,
                            }
                        if len(ratios) == 2:
                            weighted = 0.35 * ratios[0] + 0.65 * ratios[1]
                            objective = 0.80 * weighted + 0.20 * max(ratios)
                        else:
                            objective = ratios[0]
                        candidates.append(
                            {
                                "track": track_name,
                                "space": space,
                                "weight_method": weight_name,
                                "l2_strength": l2_strength,
                                "top_k": top_k,
                                "correlation_limit": correlation_limit,
                                "calibration": calibration_mode,
                                "objective": float(objective),
                                "transition_parameters": transition_results,
                            }
                        )
    return sorted(candidates, key=lambda item: item["objective"])


def fit_deployment(frame, pool, best):
    calibration = frame[frame["season"].eq(2024)]
    selected = select_diverse_models(
        calibration,
        pool,
        best["top_k"],
        best["correlation_limit"],
        salt=20_250,
    )
    x = calibration[selected].to_numpy(float)
    y = calibration[TARGET].to_numpy("int8")
    if best["weight_method"] == "equal":
        weights = np.repeat(1.0 / len(selected), len(selected))
    else:
        weights = optimize_weights(
            x,
            y,
            best["space"],
            float(best["l2_strength"]),
            salt=20_251,
        )
    raw = blend(x, weights, best["space"])
    params, _ = fit_calibrator(
        best["calibration"],
        y,
        raw,
        salt=20_252,
        groups=cohort_values(calibration, best["calibration"]),
    )
    return {
        "trained_on_oof_season": 2024,
        "selected_models": selected,
        "weights": weights.tolist(),
        "space": best["space"],
        "weight_method": best["weight_method"],
        "l2_strength": best["l2_strength"],
        "top_k": best["top_k"],
        "correlation_limit": best["correlation_limit"],
        "calibration": best["calibration"],
        "calibrator_params": params,
    }


def serializable_candidate(candidate):
    return candidate


def main():
    started = time.time()
    frame, model_columns, inventory = load_pool()
    # Final ZIP automation currently supports the V1 and enhanced seed-bagged
    # tree models. Other sources remain in the inventory/correlation analysis
    # and validation registry, but cannot silently enter an undeployable blend.
    deployable_columns = [
        column
        for column in model_columns
        if column.startswith("v1__") or column.startswith("enh__")
    ]
    robust_pool = candidate_pool(
        frame, deployable_columns, [2022, 2023, 2024]
    )
    recent_pool = candidate_pool(frame, deployable_columns, [2023, 2024])
    if len(robust_pool) < 2 or len(recent_pool) < 2:
        raise RuntimeError("OOF candidate pool is too small")

    robust = evaluate_track(
        frame, robust_pool, [(2022, 2023), (2023, 2024)], "robust"
    )
    performance = evaluate_track(
        frame, recent_pool, [(2023, 2024)], "performance"
    )
    best_by_track = {"robust": robust[0], "performance": performance[0]}
    deployment = {
        "robust": fit_deployment(frame, robust_pool, robust[0]),
        "performance": fit_deployment(frame, recent_pool, performance[0]),
    }

    prediction_rows = []
    for track, best in best_by_track.items():
        for transition, parameters in best["transition_parameters"].items():
            validation_year = int(transition.split("_to_")[1])
            validation = frame[frame["season"].eq(validation_year)]
            matrix = validation[parameters["selected_models"]].to_numpy(float)
            raw = blend(matrix, np.asarray(parameters["weights"]), best["space"])
            prediction = apply_calibrator(
                best["calibration"],
                parameters["calibrator_params"],
                raw,
                groups=cohort_values(validation, best["calibration"]),
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "row_id": validation["row_id"].to_numpy(),
                        "season": validation_year,
                        TARGET: validation[TARGET].to_numpy("int8"),
                        "track": track,
                        "prediction": prediction.astype("float32"),
                    }
                )
            )
    pd.concat(prediction_rows, ignore_index=True).to_parquet(
        WORK_DIR / "enhanced_ensemble_oof_predictions.parquet", index=False
    )

    all_candidates = robust + performance
    flat_rows = []
    for candidate in all_candidates:
        row = {
            key: value
            for key, value in candidate.items()
            if key != "transition_parameters"
        }
        for transition, metrics in candidate["transition_parameters"].items():
            row[f"{transition}_bss"] = metrics["bss"]
            row[f"{transition}_normalized_brier"] = metrics["normalized_brier"]
            row[f"{transition}_model_count"] = len(metrics["selected_models"])
        flat_rows.append(row)
    pd.DataFrame(flat_rows).sort_values(["track", "objective"]).to_csv(
        WORK_DIR / "enhanced_ensemble_candidates.csv", index=False
    )

    valid_2024 = frame[frame["season"].eq(2024)]
    correlation_models = recent_pool[:]
    sample = deterministic_sample(len(valid_2024), 100_000, 20_240)
    residuals = (
        valid_2024.iloc[sample][correlation_models].to_numpy(float)
        - valid_2024.iloc[sample][TARGET].to_numpy(float)[:, None]
    )
    pd.DataFrame(
        np.corrcoef(residuals, rowvar=False),
        index=correlation_models,
        columns=correlation_models,
    ).to_csv(WORK_DIR / "enhanced_residual_correlation_2024.csv")

    result = {
        "created_at_epoch": time.time(),
        "elapsed_sec": time.time() - started,
        "inventory": inventory,
        "deployable_prefixes": ["v1__", "enh__"],
        "pool_sizes": {"robust": len(robust_pool), "performance": len(recent_pool)},
        "robust_pool": robust_pool,
        "performance_pool": recent_pool,
        "best": {
            track: serializable_candidate(candidate)
            for track, candidate in best_by_track.items()
        },
        "deployment": deployment,
        "top_20": {
            "robust": [serializable_candidate(item) for item in robust[:20]],
            "performance": [serializable_candidate(item) for item in performance[:20]],
        },
        "validation_rules": {
            "robust": "select/fit on 2022 then evaluate 2023; select/fit on 2023 then evaluate 2024",
            "performance": "select/fit on 2023 then evaluate 2024",
            "deployment": "refit meta weights and calibrator on 2024 OOF only",
        },
    }
    (WORK_DIR / "enhanced_ensemble_selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best": result["best"],
                "deployment": deployment,
                "pool_sizes": result["pool_sizes"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
