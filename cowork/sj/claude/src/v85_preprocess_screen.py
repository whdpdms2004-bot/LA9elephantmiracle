"""strict F1 위에서 피처군별 전처리를 넓게 Val2024 GPU screen한다.

모든 통계 기반 변환은 fold 학습행에서만 fit한다. validation/test 행들끼리의
집계·분포·순위는 사용하지 않는다. screen은 후보 축소용이며 상위 조합은
2022/2023/2024 full confirm을 별도로 수행한다.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


HERE = Path(__file__).resolve().parent
SJ = HERE.parents[1]
# 2026-08-18 feature_campaign_1000 -> claude/src 이관.
# 데이터/산출물은 캠페인 폴더에 그대로 있으므로 CAMPAIGN 으로 가리킨다.
CAMPAIGN = SJ / "feature_campaign_1000"
MODEL_OPT = SJ / "experiment" / "model_optimization"
sys.path.insert(0, str(MODEL_OPT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CAMPAIGN))

from evaluate_bucketed_residual import logit, sigmoid
from evaluate_train_only_season_offsets import forecast_offset
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    TARGET,
    probability_metrics,
    recency_weights,
)
from v77_single_xgb_screen import (
    build_component_unique,
    build_component_unique_forward,
)
from v80_single_catboost import make_features, raw_bss


FOLD = 2024
SEED = 20262844
PARAMS_PATH = MODEL_OPT / "catboost_v2r200_tm500_robust_best.json"
OUTPUT = CAMPAIGN / "outputs" / "preprocess_screen"
ID_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]
COUNT_COLUMNS = ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]
RATE_SPECS = [
    ("pitcher_success", "asof_pitcher_success_rate", "asof_pitcher_n"),
    ("pitcher_reverse", "asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("pitcher_middle", "asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("pitcher_ball", "asof_pitcher_ball_rate", "asof_pitcher_n"),
    ("pitcher_strike", "asof_pitcher_strike_rate", "asof_pitcher_n"),
    ("batter_success", "asof_batter_success_rate", "asof_batter_n"),
    ("batter_middle", "asof_batter_middle_rate", "asof_batter_n"),
    ("pitcher_fastball", "asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n"),
    ("pitcher_breaking", "asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n"),
    ("pitcher_offspeed", "asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"),
]
PREV_SUCCESS = [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
PREV_MIDDLE = [f"asof_pitcher_prev{k}_game_middle_rate" for k in (1, 3, 5)]
NOMINAL_CATEGORICAL = {
    "top_bottom", "game_type", "base_state",
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "runner_out_state",
    "handedness_matchup", "game_dayofweek",
}
DEFAULT_ARMS = [
    "baseline", "ordinal_numeric", "drop_ids", "id_frequency",
    "rate_multiscale", "rate_geometry", "count_multiscale", "recent_shape",
    "temporal_cyclic", "context_robust", "trackman_quality",
    "trackman_compact", "component_shape", "component_compact",
    "no_trackman", "no_component", "all_additive", "all_compact",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--depth", type=int, default=8)
    return parser.parse_args()


def add_columns(frame: pd.DataFrame, values: dict[str, np.ndarray]) -> pd.DataFrame:
    if not values:
        return frame
    extra = pd.DataFrame(values, index=frame.index)
    return pd.concat([frame, extra], axis=1)


def id_frequency(frame: pd.DataFrame, train_mask: pd.Series):
    extra = {}
    for column in ID_COLUMNS:
        counts = frame.loc[train_mask, column].value_counts(dropna=False)
        frequency = frame[column].map(counts).fillna(0).to_numpy(np.float64)
        extra[f"prep_{column}_log_frequency"] = np.log1p(frequency)
        extra[f"prep_{column}_unseen"] = (frequency == 0).astype(np.int8)
    return extra


def rate_multiscale(frame: pd.DataFrame, train_mask: pd.Series):
    extra = {}
    for name, rate_column, count_column in RATE_SPECS:
        rate = pd.to_numeric(frame[rate_column], errors="coerce").to_numpy(np.float64)
        count = np.nan_to_num(
            pd.to_numeric(frame[count_column], errors="coerce").to_numpy(np.float64),
            nan=0.0,
        )
        prior = float(pd.to_numeric(
            frame.loc[train_mask, rate_column], errors="coerce").median())
        filled = np.where(np.isfinite(rate), rate, prior)
        for strength in (50.0, 500.0, 1000.0):
            tag = int(strength)
            extra[f"prep_{name}_smooth_{tag}"] = (
                count * filled + strength * prior) / (count + strength)
            extra[f"prep_{name}_rel_{tag}"] = count / (count + strength)
    return extra


def rate_geometry(frame: pd.DataFrame):
    extra = {}
    rate_columns = [column for _, column, _ in RATE_SPECS]
    for column in rate_columns + ["home_win_expectancy", "away_win_expectancy"]:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        extra[f"prep_logit_{column}"] = logit(values)
    pitcher = logit(frame["asof_pitcher_success_rate"].to_numpy(np.float64))
    batter = logit(frame["asof_batter_success_rate"].to_numpy(np.float64))
    extra["prep_success_logit_gap"] = pitcher - batter
    extra["prep_expectancy_logit_gap"] = (
        logit(frame["home_win_expectancy"].to_numpy(np.float64))
        - logit(frame["away_win_expectancy"].to_numpy(np.float64)))
    mix = np.column_stack([
        frame["asof_pitcher_fastball_rate"].to_numpy(np.float64),
        frame["asof_pitcher_breaking_rate"].to_numpy(np.float64),
        frame["asof_pitcher_offspeed_rate"].to_numpy(np.float64),
    ])
    mix = np.clip(mix, 1e-6, 1.0)
    mix_sum = np.nansum(mix, axis=1, keepdims=True)
    normalized = mix / np.where(mix_sum > 0, mix_sum, np.nan)
    extra["prep_pitchmix_entropy"] = -np.nansum(
        normalized * np.log(normalized), axis=1)
    extra["prep_pitchmix_concentration"] = np.nansum(normalized ** 2, axis=1)
    extra["prep_fastball_breaking_logratio"] = np.log(mix[:, 0] / mix[:, 1])
    extra["prep_fastball_offspeed_logratio"] = np.log(mix[:, 0] / mix[:, 2])
    return extra


def count_multiscale(frame: pd.DataFrame):
    extra = {}
    bins = [-np.inf, 0, 25, 100, 500, 1000, 2000, 4000, np.inf]
    categorical = []
    for column in COUNT_COLUMNS:
        values = np.nan_to_num(
            pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64),
            nan=0.0,
        )
        extra[f"prep_sqrt_{column}"] = np.sqrt(np.clip(values, 0, None))
        extra[f"prep_{column}_bucket"] = pd.cut(
            values, bins=bins, labels=False, include_lowest=True).astype(str)
        categorical.append(f"prep_{column}_bucket")
        for strength in (25.0, 100.0, 500.0, 2000.0):
            extra[f"prep_{column}_rel_{int(strength)}"] = (
                values / (values + strength))
    return extra, categorical


def recent_shape(frame: pd.DataFrame):
    extra = {}
    for tag, columns, career in (
        ("success", PREV_SUCCESS, "asof_pitcher_success_rate"),
        ("middle", PREV_MIDDLE, "asof_pitcher_middle_rate"),
    ):
        values = np.column_stack([
            pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
            for column in columns
        ])
        career_values = frame[career].to_numpy(np.float64)
        extra[f"prep_recent_{tag}_weighted"] = np.nansum(
            values * np.array([5.0, 3.0, 1.0]), axis=1) / np.nansum(
                np.isfinite(values) * np.array([5.0, 3.0, 1.0]), axis=1)
        extra[f"prep_recent_{tag}_slope"] = values[:, 0] - values[:, 2]
        extra[f"prep_recent_{tag}_curvature"] = (
            values[:, 0] - 2.0 * values[:, 1] + values[:, 2])
        extra[f"prep_recent_{tag}_shock"] = values[:, 0] - career_values
        extra[f"prep_recent_{tag}_abs_shock"] = np.abs(
            values[:, 0] - career_values)
        extra[f"prep_recent_{tag}_missing"] = np.isnan(values).sum(axis=1)
    return extra


def temporal_cyclic(frame: pd.DataFrame, fold: int):
    month = frame["game_month"].to_numpy(np.float64)
    day = frame["game_dayofweek"].to_numpy(np.float64)
    inning = frame["inning"].to_numpy(np.float64)
    return {
        "prep_month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
        "prep_month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
        "prep_day_sin": np.sin(2 * np.pi * day / 7.0),
        "prep_day_cos": np.cos(2 * np.pi * day / 7.0),
        "prep_inning_clipped": np.minimum(inning, 10.0),
        "prep_inning_extra": np.maximum(inning - 9.0, 0.0),
        "prep_years_to_prediction": frame["season"].to_numpy(np.float64) - fold,
        "prep_season_month_progress": (
            (frame["season"].to_numpy(np.float64) - fold) * 12.0 + month),
    }


def context_robust(frame: pd.DataFrame):
    extra = {}
    for column in (
        "run_top_before", "run_bot_before", "run_total_before",
        "score_diff_home", "score_diff_pitcher_team",
    ):
        values = frame[column].to_numpy(np.float64)
        extra[f"prep_signed_log_{column}"] = np.sign(values) * np.log1p(
            np.abs(values))
    leverage = np.clip(frame["li"].to_numpy(np.float64), 0, None)
    extra["prep_log1p_li"] = np.log1p(leverage)
    extra["prep_li_capped_3"] = np.minimum(leverage, 3.0)
    extra["prep_expectancy_centered"] = (
        frame["home_win_expectancy"].to_numpy(np.float64) - 0.5)
    extra["prep_runner_pressure"] = (
        frame["runner_on_1b"].to_numpy(np.float64)
        + 2.0 * frame["runner_on_2b"].to_numpy(np.float64)
        + 3.0 * frame["runner_on_3b"].to_numpy(np.float64))
    extra["prep_outs_remaining"] = 3.0 - frame["outs_before"].to_numpy(np.float64)
    return extra


def robust_z(frame: pd.DataFrame, train_mask: pd.Series, columns: list[str]):
    arrays = []
    for column in columns:
        train = pd.to_numeric(
            frame.loc[train_mask, column], errors="coerce").to_numpy(np.float64)
        median = float(np.nanmedian(train))
        q25, q75 = np.nanpercentile(train, [25, 75])
        scale = max(float(q75 - q25), 1e-6)
        values = pd.to_numeric(
            frame[column], errors="coerce").to_numpy(np.float64)
        arrays.append((values - median) / scale)
    return np.column_stack(arrays)


def trackman_quality(frame: pd.DataFrame, train_mask: pd.Series):
    extra = {}
    tm_columns = [column for column in frame if column.startswith(("tm500_", "cw_"))]
    physical = [
        column for column in tm_columns
        if column.startswith("tm500_latest_") and column.endswith("_mean")]
    dispersion = [
        column for column in tm_columns
        if column.startswith("tm500_latest_") and column.endswith("_std")]
    shifts = [column for column in tm_columns if column.endswith("_minus_recent")]
    raw_matrix = frame[tm_columns].apply(
        pd.to_numeric, errors="coerce").to_numpy(np.float64)
    extra["prep_tm_missing_count"] = np.isnan(raw_matrix).sum(axis=1)
    extra["prep_tm_missing_ratio"] = np.isnan(raw_matrix).mean(axis=1)
    if physical:
        values = robust_z(frame, train_mask, physical)
        extra["prep_tm_style_l2"] = np.sqrt(np.nanmean(values ** 2, axis=1))
        extra["prep_tm_style_mean"] = np.nanmean(values, axis=1)
        extra["prep_tm_style_std"] = np.nanstd(values, axis=1)
    if dispersion:
        values = robust_z(frame, train_mask, dispersion)
        extra["prep_tm_dispersion_mean"] = np.nanmean(values, axis=1)
        extra["prep_tm_dispersion_l2"] = np.sqrt(np.nanmean(values ** 2, axis=1))
    if shifts:
        values = robust_z(frame, train_mask, shifts)
        extra["prep_tm_shift_l2"] = np.sqrt(np.nanmean(values ** 2, axis=1))
        extra["prep_tm_shift_mean"] = np.nanmean(values, axis=1)
    eligible = np.clip(frame["tm500_eligible_seasons"].to_numpy(np.float64), 0, None)
    total = np.clip(frame["tm500_total_pitches"].to_numpy(np.float64), 0, None)
    extra["prep_tm_pitches_per_season"] = total / np.maximum(eligible, 1.0)
    extra["prep_tm_crosswalk_balance"] = (
        np.log1p(frame["cw_total_main_n"].to_numpy(np.float64))
        - np.log1p(frame["cw_total_trackman_n"].to_numpy(np.float64)))
    return extra


def component_shape(frame: pd.DataFrame):
    extra = {}
    split_columns = [
        column for column in frame
        if column.startswith("sx_cf_") and column.endswith("_split")]
    rel_columns = [
        column for column in frame
        if column.startswith("sx_cf_") and column.endswith("_rel")]
    for column in split_columns:
        values = frame[column].to_numpy(np.float64)
        extra[f"prep_abs_{column}"] = np.abs(values)
        extra[f"prep_sign_{column}"] = np.sign(values).astype(np.int8)
    if split_columns:
        values = frame[split_columns].to_numpy(np.float64)
        extra["prep_component_abs_sum"] = np.nansum(np.abs(values), axis=1)
        extra["prep_component_abs_max"] = np.nanmax(np.abs(values), axis=1)
    if rel_columns:
        values = frame[rel_columns].to_numpy(np.float64)
        extra["prep_component_rel_mean"] = np.nanmean(values, axis=1)
        extra["prep_component_rel_min"] = np.nanmin(values, axis=1)
    return extra


def compact_trackman(features: list[str]) -> list[str]:
    drop = {
        column for column in features
        if column.startswith("cw_")
        or "_between_" in column
        or (column.startswith("tm500_recent_") and
            f"{column.replace('tm500_recent_', 'tm500_latest_')}_minus_recent" in features)
    }
    return [column for column in features if column not in drop]


def compact_component(features: list[str]) -> list[str]:
    drop = {
        column for column in features
        if column.startswith("sx_cf_") and column.endswith("_split")
    }
    return [column for column in features if column not in drop]


def build_arm(base: pd.DataFrame, base_features: list[str], arm: str,
              train_mask: pd.Series, fold: int):
    frame = base
    features = list(base_features)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    extras = {}
    extra_categorical = []

    def apply(name: str):
        nonlocal features, categorical, extras, extra_categorical
        if name == "ordinal_numeric":
            categorical = [
                column for column in categorical if column in NOMINAL_CATEGORICAL]
        elif name == "drop_ids":
            features = [column for column in features if column not in ID_COLUMNS]
            categorical = [column for column in categorical if column not in ID_COLUMNS]
        elif name == "id_frequency":
            apply("drop_ids")
            extras.update(id_frequency(frame, train_mask))
        elif name == "rate_multiscale":
            extras.update(rate_multiscale(frame, train_mask))
        elif name == "rate_geometry":
            extras.update(rate_geometry(frame))
        elif name == "count_multiscale":
            values, cats = count_multiscale(frame)
            extras.update(values)
            extra_categorical.extend(cats)
        elif name == "recent_shape":
            extras.update(recent_shape(frame))
        elif name == "temporal_cyclic":
            extras.update(temporal_cyclic(frame, fold))
        elif name == "context_robust":
            extras.update(context_robust(frame))
        elif name == "trackman_quality":
            extras.update(trackman_quality(frame, train_mask))
        elif name == "trackman_compact":
            features = compact_trackman(features)
        elif name == "component_shape":
            extras.update(component_shape(frame))
        elif name == "component_compact":
            features = compact_component(features)
        elif name == "no_trackman":
            features = [
                column for column in features
                if not column.startswith(("tm500_", "cw_"))]
        elif name == "no_component":
            features = [
                column for column in features if not column.startswith("sx_cf_")]
        elif name == "baseline":
            pass
        else:
            raise ValueError(name)

    if arm == "all_additive":
        for name in (
            "rate_multiscale", "rate_geometry", "count_multiscale",
            "recent_shape", "temporal_cyclic", "context_robust",
            "trackman_quality", "component_shape",
        ):
            apply(name)
    elif arm == "all_compact":
        for name in (
            "ordinal_numeric", "id_frequency", "rate_multiscale",
            "rate_geometry", "count_multiscale", "recent_shape",
            "temporal_cyclic", "context_robust", "trackman_quality",
            "component_shape", "trackman_compact", "component_compact",
        ):
            apply(name)
    else:
        apply(arm)

    frame = add_columns(frame, extras)
    features.extend(extras)
    features = list(dict.fromkeys(features))
    categorical = list(dict.fromkeys(
        [column for column in categorical + extra_categorical if column in features]))
    return frame, features, categorical


def main() -> None:
    args = parse_args()
    arms = [value for value in args.arms.split(",") if value]
    unknown = [arm for arm in arms if arm not in DEFAULT_ARMS]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, enhanced_features = load_enhanced_frame()
    train_mask = frame["season"].lt(FOLD)
    valid_mask = frame["season"].eq(FOLD)
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    static = build_component_unique(frame, enhanced_features, FOLD)
    forward = build_component_unique_forward(
        frame, enhanced_features, FOLD, cache={FOLD: static})
    base, f1_features = make_features(
        frame, enhanced_features, FOLD, "F1", forward)
    rates = frame.groupby("season")[TARGET].mean()
    offset = forecast_offset(rates, FOLD, window=None, damping=0.25)
    best = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(best.pop("half_life"))
    best.update({
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "border_count": 128,
    })
    weights = recency_weights(frame.loc[train_mask, "season"], FOLD, half_life)
    del forward, static
    gc.collect()
    print(
        f"prepared fold={FOLD} train={int(train_mask.sum())} "
        f"valid={int(valid_mask.sum())} base_features={len(f1_features)} "
        f"arms={len(arms)} offset={offset:+.8f}", flush=True)

    rows = []
    for arm in arms:
        started = time.time()
        arm_frame, features, categorical = build_arm(
            base, f1_features, arm, train_mask, FOLD)
        model_frame = arm_frame[features].copy()
        for column in categorical:
            model_frame[column] = (
                model_frame[column].fillna("__MISSING__").astype(str))
        train_pool = Pool(
            model_frame.loc[train_mask],
            label=frame.loc[train_mask, TARGET],
            cat_features=categorical,
            weight=weights,
        )
        valid_pool = Pool(
            model_frame.loc[valid_mask],
            label=valid_y,
            cat_features=categorical,
        )
        model = CatBoostClassifier(
            **best,
            loss_function="Logloss",
            eval_metric="Logloss",
            task_type="GPU",
            devices="0",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=120,
        )
        raw_prediction = model.predict_proba(valid_pool)[:, 1].astype(np.float64)
        prediction = sigmoid(logit(raw_prediction) + offset)
        raw_score = probability_metrics(valid_y, raw_prediction)
        adjusted_score = probability_metrics(valid_y, prediction)
        elapsed = time.time() - started
        np.save(OUTPUT / f"{arm}_{FOLD}.npy", raw_prediction)
        rows.append({
            "arm": arm,
            "fold": FOLD,
            "n_features": len(features),
            "n_categorical": len(categorical),
            "best_iteration": int(model.get_best_iteration()),
            "elapsed_sec": elapsed,
            "bss_raw": raw_bss(raw_score),
            "bss_adjusted": raw_bss(adjusted_score),
            "brier_adjusted": adjusted_score["brier"],
            "pred_mean_adjusted": adjusted_score["pred_mean"],
        })
        print(
            f"{arm:20s} f={len(features):3d} cat={len(categorical):2d} "
            f"raw={rows[-1]['bss_raw']:8.3f} adj={rows[-1]['bss_adjusted']:8.3f} "
            f"iter={rows[-1]['best_iteration']:4d} t={elapsed:6.1f}s",
            flush=True,
        )
        del model, train_pool, valid_pool, model_frame, arm_frame
        gc.collect()

    result = pd.DataFrame(rows)
    previous = OUTPUT / "metrics.csv"
    if previous.is_file():
        old = pd.read_csv(previous)
        result = pd.concat([old, result], ignore_index=True)
        result = result.drop_duplicates(["arm", "fold"], keep="last")
    baseline = float(result.loc[result["arm"].eq("baseline"), "bss_adjusted"].iloc[-1]) \
        if result["arm"].eq("baseline").any() else np.nan
    result["delta_vs_baseline"] = result["bss_adjusted"] - baseline
    result = result.sort_values("bss_adjusted", ascending=False)
    result.to_csv(previous, index=False)
    print("\n" + result.round(4).to_string(index=False))
    print(f"saved -> {previous}")


if __name__ == "__main__":
    main()
