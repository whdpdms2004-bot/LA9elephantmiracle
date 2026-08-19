"""3WAY final submission의 추론 전용 추가 피처 계산기.

통계값과 ID 빈도는 2019~2024 학습 데이터에서 미리 고정해 전달받는다.
평가 프레임에서는 각 행의 값과 고정 자산만 사용한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def _id_frequency(frame: pd.DataFrame, lookups: dict[str, pd.Series]) -> dict:
    extra = {}
    for column in ID_COLUMNS:
        frequency = frame[column].map(lookups[column]).fillna(0).to_numpy(np.float64)
        extra[f"prep_{column}_log_frequency"] = np.log1p(frequency)
        extra[f"prep_{column}_unseen"] = (frequency == 0).astype(np.int8)
    return extra


def _rate_multiscale(frame: pd.DataFrame, spec: dict) -> dict:
    extra = {}
    priors = spec["rate_priors"]
    for name, rate_column, count_column in RATE_SPECS:
        rate = _numeric(frame, rate_column)
        count = np.nan_to_num(_numeric(frame, count_column), nan=0.0)
        prior = float(priors[rate_column])
        filled = np.where(np.isfinite(rate), rate, prior)
        for strength in (50.0, 500.0, 1000.0):
            tag = int(strength)
            extra[f"prep_{name}_smooth_{tag}"] = (
                count * filled + strength * prior) / (count + strength)
            extra[f"prep_{name}_rel_{tag}"] = count / (count + strength)
    return extra


def _count_multiscale(frame: pd.DataFrame) -> dict:
    extra = {}
    bins = [-np.inf, 0, 25, 100, 500, 1000, 2000, 4000, np.inf]
    for column in COUNT_COLUMNS:
        values = np.nan_to_num(_numeric(frame, column), nan=0.0)
        extra[f"prep_sqrt_{column}"] = np.sqrt(np.clip(values, 0, None))
        extra[f"prep_{column}_bucket"] = pd.cut(
            values, bins=bins, labels=False, include_lowest=True).astype(str)
        for strength in (25.0, 100.0, 500.0, 2000.0):
            extra[f"prep_{column}_rel_{int(strength)}"] = (
                values / (values + strength))
    return extra


def _temporal_cyclic(frame: pd.DataFrame, prediction_season: int) -> dict:
    month = _numeric(frame, "game_month")
    day = _numeric(frame, "game_dayofweek")
    inning = _numeric(frame, "inning")
    season = _numeric(frame, "season")
    return {
        "prep_month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
        "prep_month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
        "prep_day_sin": np.sin(2 * np.pi * day / 7.0),
        "prep_day_cos": np.cos(2 * np.pi * day / 7.0),
        "prep_inning_clipped": np.minimum(inning, 10.0),
        "prep_inning_extra": np.maximum(inning - 9.0, 0.0),
        "prep_years_to_prediction": season - prediction_season,
        "prep_season_month_progress": (
            (season - prediction_season) * 12.0 + month),
    }


def _robust_values(frame: pd.DataFrame, columns: list[str], spec: dict) -> np.ndarray:
    arrays = []
    for column in columns:
        values = _numeric(frame, column)
        stats = spec["robust_stats"][column]
        arrays.append((values - float(stats["median"])) / float(stats["scale"]))
    return np.column_stack(arrays)


def _trackman_quality(frame: pd.DataFrame, spec: dict) -> dict:
    extra = {}
    quality = spec["trackman_quality"]
    tm_columns = quality["tm_columns"]
    raw_matrix = frame[tm_columns].apply(
        pd.to_numeric, errors="coerce").to_numpy(np.float64)
    extra["prep_tm_missing_count"] = np.isnan(raw_matrix).sum(axis=1)
    extra["prep_tm_missing_ratio"] = np.isnan(raw_matrix).mean(axis=1)
    physical = quality["physical"]
    dispersion = quality["dispersion"]
    shifts = quality["shifts"]
    if physical:
        values = _robust_values(frame, physical, quality)
        extra["prep_tm_style_l2"] = np.sqrt(np.nanmean(values ** 2, axis=1))
        extra["prep_tm_style_mean"] = np.nanmean(values, axis=1)
        extra["prep_tm_style_std"] = np.nanstd(values, axis=1)
    if dispersion:
        values = _robust_values(frame, dispersion, quality)
        extra["prep_tm_dispersion_mean"] = np.nanmean(values, axis=1)
        extra["prep_tm_dispersion_l2"] = np.sqrt(np.nanmean(values ** 2, axis=1))
    if shifts:
        values = _robust_values(frame, shifts, quality)
        extra["prep_tm_shift_l2"] = np.sqrt(np.nanmean(values ** 2, axis=1))
        extra["prep_tm_shift_mean"] = np.nanmean(values, axis=1)
    eligible = np.clip(_numeric(frame, "tm500_eligible_seasons"), 0, None)
    total = np.clip(_numeric(frame, "tm500_total_pitches"), 0, None)
    extra["prep_tm_pitches_per_season"] = total / np.maximum(eligible, 1.0)
    extra["prep_tm_crosswalk_balance"] = (
        np.log1p(_numeric(frame, "cw_total_main_n"))
        - np.log1p(_numeric(frame, "cw_total_trackman_n")))
    return extra


def add_target_features(
        frame: pd.DataFrame,
        combo: list[str],
        spec: dict,
        id_lookups: dict[str, pd.Series],
        prediction_season: int = 2025) -> pd.DataFrame:
    """고정 학습 자산과 한 행의 입력만으로 타깃별 추가 열을 만든다."""
    extra = {}
    for name in combo:
        if name == "id_frequency":
            extra.update(_id_frequency(frame, id_lookups))
        elif name == "rate_multiscale":
            extra.update(_rate_multiscale(frame, spec))
        elif name == "count_multiscale":
            extra.update(_count_multiscale(frame))
        elif name == "temporal_cyclic":
            extra.update(_temporal_cyclic(frame, prediction_season))
        elif name == "trackman_quality":
            extra.update(_trackman_quality(frame, spec))
        elif name in {"drop_ids", "no_trackman"}:
            continue
        else:
            raise ValueError(f"unsupported final transform: {name}")
    if not extra:
        return frame
    return pd.concat([frame, pd.DataFrame(extra, index=frame.index)], axis=1)
