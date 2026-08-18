"""F1 제출 모델의 추론 전용 성분 피처 변환기.

학습 데이터로 미리 고정한 spec과 lookup만 사용하며, 평가 행들 사이의
집계나 상태 공유는 하지 않는다. 이 파일에는 학습/집계 함수가 없다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


TARGET = "control_success"
RATES = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
PREV_S = [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
PREV_M = [f"asof_pitcher_prev{k}_game_middle_rate" for k in (1, 3, 5)]
CAT_COLS = ["top_bottom", "game_type", "base_state"]
RATE_SPECS = [
    ("pitcher_success", "asof_pitcher_success_rate", "asof_pitcher_n"),
    ("pitcher_reverse", "asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("pitcher_middle", "asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("batter_success", "asof_batter_success_rate", "asof_batter_n"),
    ("batter_middle", "asof_batter_middle_rate", "asof_batter_n"),
]
BAT_COMPONENTS = ["m", "r", "mr", "ob", "oz"]


def count_bucket(frame: pd.DataFrame) -> np.ndarray:
    """볼카운트 3군: 투수우세(0), 중립(1), 타자우세(2)."""
    balls = pd.to_numeric(frame["balls_before"]).to_numpy()
    strikes = pd.to_numeric(frame["strikes_before"]).to_numpy()
    return np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))


def inning_bucket(frame: pd.DataFrame) -> np.ndarray:
    """이닝 4군: 1-3, 4-6, 7-9, 연장."""
    return np.digitize(pd.to_numeric(frame["inning"]).to_numpy(), [4, 7, 10])


def build(frame: pd.DataFrame, spec: dict, platoon: pd.DataFrame,
          bat_platoon: pd.DataFrame | None = None,
          count_platoon: pd.DataFrame | None = None,
          inning_platoon: pd.DataFrame | None = None) -> pd.DataFrame:
    """각 입력 행만으로 피처를 만들고 고정된 학습 lookup을 연결한다."""
    priors = spec["priors"]
    strength = float(spec["strength"])
    lam = float(spec["lam_prof"])
    out = {}

    for column in frame.columns:
        if column in ("row_id", TARGET):
            continue
        if column in CAT_COLS:
            out[column] = (
                frame[column].astype(str).map(spec["cat_map"][column])
                .fillna(-1).to_numpy(np.float64))
        else:
            out[column] = pd.to_numeric(
                frame[column], errors="coerce").to_numpy(np.float64)

    balls = out["balls_before"]
    strikes = out["strikes_before"]
    pitcher_hand = out["pitcher_hand"]
    batter_hand = out["batter_hand"]
    pitcher_n = out["asof_pitcher_n"]
    out["count_state"] = balls * 3 + strikes
    out["handedness_matchup"] = pitcher_hand * 2 + batter_hand
    out["runner_out_state"] = out["num_runners_on"] * 3 + out["outs_before"]
    out["score_abs"] = np.abs(out["score_diff_pitcher_team"])
    out["late_inning"] = (out["inning"] >= 7).astype(np.float64)
    out["high_leverage"] = (out["li"] >= 2).astype(np.float64)
    out["log1p_asof_pitcher_n"] = np.log1p(pitcher_n)
    out["log1p_asof_batter_n"] = np.log1p(out["asof_batter_n"])
    for window in (1, 3, 5):
        out[f"pitcher_success_delta_prev{window}"] = (
            out[f"asof_pitcher_prev{window}_game_success_rate"]
            - out["asof_pitcher_success_rate"])
        out[f"pitcher_middle_delta_prev{window}"] = (
            out[f"asof_pitcher_prev{window}_game_middle_rate"]
            - out["asof_pitcher_middle_rate"])
    out["ball_strike_gap"] = (
        out["asof_pitcher_ball_rate"] - out["asof_pitcher_strike_rate"])
    for name, rate_column, n_column in RATE_SPECS:
        n = out[n_column]
        rate = np.where(
            np.isnan(out[rate_column]), priors[name], out[rate_column])
        out[f"{name}_is_missing"] = np.isnan(
            out[rate_column]).astype(np.float64)
        out[f"{name}_smoothed"] = (
            n * rate + strength * priors[name]) / (n + strength)
        out[f"{name}_reliability"] = n / (n + strength)

    for column in RATES:
        median = spec["rate_median"][column]
        rate = np.where(np.isnan(out[column]), median, out[column])
        out[f"prof200_{column}"] = (
            pitcher_n * rate + lam * median) / (pitcher_n + lam)

    previous = {
        column: np.where(
            np.isnan(out[column]), spec["prev_median"][column], out[column])
        for column in PREV_S + PREV_M
    }
    out["prev_trend_s"] = previous[PREV_S[0]] - previous[PREV_S[2]]
    out["prev_trend_m"] = previous[PREV_M[0]] - previous[PREV_M[2]]
    out["prev_std_s"] = np.std(
        np.vstack([previous[column] for column in PREV_S]), axis=0)
    out["prev_std_m"] = np.std(
        np.vstack([previous[column] for column in PREV_M]), axis=0)
    out["prev_miss_cnt"] = sum(
        np.isnan(out[column]).astype(np.float64)
        for column in PREV_S + PREV_M)
    for index, (success_column, middle_column) in enumerate(
            zip(PREV_S, PREV_M)):
        out[f"faildir_{index}"] = (
            previous[middle_column] - (1 - previous[success_column]))
    out["rel200"] = pitcher_n / (pitcher_n + lam)

    pitcher_key = pd.MultiIndex.from_arrays([
        pd.to_numeric(frame["pitcher_id"]),
        pd.to_numeric(frame["batter_hand"]),
    ])
    pitcher_lookup = (
        platoon.set_index(["pitcher_id", "batter_hand"]).reindex(pitcher_key))
    split = pitcher_lookup["platoon_split"].fillna(0.0).to_numpy(np.float64)
    reliability = pitcher_lookup["platoon_rel"].fillna(0.0).to_numpy(np.float64)
    out["platoon_split"] = split
    out["platoon_rel"] = reliability
    out["platoon_split_w"] = split * reliability

    if bat_platoon is not None:
        batter_key = pd.MultiIndex.from_arrays([
            pd.to_numeric(frame["batter_id"]),
            pd.to_numeric(frame["pitcher_hand"]),
        ])
        batter_lookup = bat_platoon.set_index(
            ["batter_id", "pitcher_hand"]).reindex(batter_key)
        split = batter_lookup["bat_platoon_split"].fillna(0.0).to_numpy(np.float64)
        reliability = batter_lookup["bat_platoon_rel"].fillna(0.0).to_numpy(np.float64)
        out["bat_platoon_split"] = split
        out["bat_platoon_rel"] = reliability
        out["bat_platoon_split_w"] = split * reliability
        for tag in BAT_COMPONENTS:
            out[f"bat_pl_{tag}"] = (
                batter_lookup[f"bat_pl_{tag}"].fillna(0.0).to_numpy(np.float64))

    if count_platoon is not None:
        count_key = pd.MultiIndex.from_arrays([
            pd.to_numeric(frame["pitcher_id"]),
            pd.to_numeric(frame["batter_hand"]),
            count_bucket(frame),
        ])
        count_lookup = count_platoon.set_index(
            ["pitcher_id", "batter_hand", "count_bucket"]).reindex(count_key)
        split = count_lookup[
            "count_platoon_split"].fillna(0.0).to_numpy(np.float64)
        reliability = count_lookup[
            "count_platoon_rel"].fillna(0.0).to_numpy(np.float64)
        out["count_platoon_split"] = split
        out["count_platoon_rel"] = reliability
        out["count_platoon_w"] = split * reliability

    if inning_platoon is not None:
        inning_key = pd.MultiIndex.from_arrays([
            pd.to_numeric(frame["pitcher_id"]),
            pd.to_numeric(frame["batter_hand"]),
            inning_bucket(frame),
        ])
        inning_lookup = inning_platoon.set_index(
            ["pitcher_id", "batter_hand", "inning_bucket"]).reindex(inning_key)
        split = inning_lookup[
            "inning_platoon_split"].fillna(0.0).to_numpy(np.float64)
        reliability = inning_lookup[
            "inning_platoon_rel"].fillna(0.0).to_numpy(np.float64)
        out["inning_platoon_split"] = split
        out["inning_platoon_rel"] = reliability
        out["inning_platoon_w"] = split * reliability

    return pd.DataFrame(out)
