"""F1 제출 모델의 추론 전용 기본·TrackMan 피처 변환기.

평가 행 자체와 2019~2024 데이터로 고정한 투수 lookup만 사용한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
ROW_RATE_SPECS = {
    "pitcher_success": ("asof_pitcher_success_rate", "asof_pitcher_n", 0.50),
    "pitcher_reverse": ("asof_pitcher_reverse_rate", "asof_pitcher_n", 0.23),
    "pitcher_middle": ("asof_pitcher_middle_rate", "asof_pitcher_n", 0.15),
    "pitcher_ball": ("asof_pitcher_ball_rate", "asof_pitcher_n", 0.50),
    "pitcher_strike": ("asof_pitcher_strike_rate", "asof_pitcher_n", 0.50),
    "batter_success": ("asof_batter_success_rate", "asof_batter_n", 0.50),
    "batter_middle": ("asof_batter_middle_rate", "asof_batter_n", 0.15),
    "pitcher_fastball": (
        "asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n", 0.50),
    "pitcher_breaking": (
        "asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n", 0.30),
    "pitcher_offspeed": (
        "asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n", 0.20),
}


def add_v1_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["count_state"] = (
        output["balls_before"].astype(str) + "-"
        + output["strikes_before"].astype(str))
    output["runner_out_state"] = (
        output["base_state"].astype(str) + "_o"
        + output["outs_before"].astype(str))
    output["handedness_matchup"] = (
        output["pitcher_hand"].astype(str) + "_"
        + output["batter_hand"].astype(str))
    output["score_abs"] = output["score_diff_pitcher_team"].abs()
    output["late_inning"] = (output["inning"] >= 7).astype("int8")
    output["high_leverage"] = (output["li"] >= 2.0).astype("int8")
    for column in (
            "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"):
        output[f"log1p_{column}"] = np.log1p(output[column].clip(lower=0))
    for window in (1, 3, 5):
        output[f"pitcher_success_delta_prev{window}"] = (
            output[f"asof_pitcher_prev{window}_game_success_rate"]
            - output["asof_pitcher_success_rate"])
        output[f"pitcher_middle_delta_prev{window}"] = (
            output[f"asof_pitcher_prev{window}_game_middle_rate"]
            - output["asof_pitcher_middle_rate"])
    output["ball_strike_rate_sum_gap"] = (
        output["asof_pitcher_ball_rate"]
        + output["asof_pitcher_strike_rate"] - 1.0)
    return output


def add_v2_row_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = add_v1_features(frame)
    for prefix, (rate_column, count_column, prior) in ROW_RATE_SPECS.items():
        rate = output[rate_column].astype(float)
        count = output[count_column].clip(lower=0).astype(float)
        output[f"{prefix}_is_missing"] = rate.isna().astype("int8")
        for strength in (10, 50, 200, 500):
            output[f"{prefix}_smoothed_{strength}"] = (
                (rate.fillna(prior) * count + prior * strength)
                / (count + strength)).astype("float32")
            output[f"{prefix}_reliability_{strength}"] = (
                count / (count + strength)).astype("float32")
    pitcher_n = output["asof_pitcher_n"]
    batter_n = output["asof_batter_n"]
    for threshold in (0, 25, 100, 500, 1000):
        operator = "eq" if threshold == 0 else "le"
        output[f"pitcher_n_{operator}_{threshold}"] = getattr(
            pitcher_n, operator)(threshold).astype("int8")
        output[f"batter_n_{operator}_{threshold}"] = getattr(
            batter_n, operator)(threshold).astype("int8")
    success_recent = output[[
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]]
    middle_recent = output[[
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]]
    output["pitcher_recent_success_mean"] = (
        success_recent.mean(axis=1).astype("float32"))
    output["pitcher_recent_success_std"] = (
        success_recent.std(axis=1).astype("float32"))
    output["pitcher_recent_success_range"] = (
        success_recent.max(axis=1) - success_recent.min(axis=1)).astype("float32")
    output["pitcher_recent_middle_mean"] = (
        middle_recent.mean(axis=1).astype("float32"))
    output["pitcher_recent_middle_std"] = (
        middle_recent.std(axis=1).astype("float32"))
    output["pitcher_recent_middle_range"] = (
        middle_recent.max(axis=1) - middle_recent.min(axis=1)).astype("float32")
    output["pitcher_failure_rate_sum"] = (
        output["asof_pitcher_reverse_rate"]
        + output["asof_pitcher_middle_rate"]
        + 1.0 - output["asof_pitcher_success_rate"]).astype("float32")
    output["pitcher_control_component_gap"] = (
        output["asof_pitcher_success_rate"]
        + output["asof_pitcher_reverse_rate"]
        + output["asof_pitcher_middle_rate"] - 1.0).astype("float32")
    return output


def enrich_trackman(frame: pd.DataFrame, tm_columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    output["tm500_log_total_pitches"] = np.log1p(output["tm500_total_pitches"])
    output["tm500_log_last_season_n"] = np.log1p(output["tm500_last_season_n"])
    output["tm500_log_cw_total_main_n"] = np.log1p(output["cw_total_main_n"])
    output["tm500_log_cw_total_trackman_n"] = np.log1p(
        output["cw_total_trackman_n"])
    for column in tm_columns:
        if "_latest_" not in column:
            continue
        recent = column.replace("_latest_", "_recent_")
        if recent in output:
            output[f"{column}_minus_recent"] = output[column] - output[recent]
    quality = output["cw_match_seasons"].ge(2) | (
        output["cw_mean_sim"].ge(0.90) & output["cw_min_margin"].ge(0.10))
    output["tm500_high_confidence"] = quality.fillna(False).astype("int8")
    output["tm500_low_confidence"] = (
        output["tm500_available"].eq(1) & ~quality.fillna(False)).astype("int8")
    return output


def build_feature_frame(test: pd.DataFrame, version: str,
                        metadata: dict) -> pd.DataFrame:
    if version != "enhanced":
        raise ValueError(f"F1 runtime only supports enhanced, got {version!r}")
    output = add_v2_row_features(test)
    lookup = pd.read_csv(MODEL_DIR / metadata["trackman_lookup_file"])
    output = output.merge(
        lookup, on="pitcher_id", how="left", validate="many_to_one")
    output["tm500_available"] = output["tm500_available"].fillna(0).astype("int8")
    output["tm500_unavailable"] = output[
        "tm500_unavailable"].fillna(1).astype("int8")
    return enrich_trackman(output, metadata["trackman_columns"])
