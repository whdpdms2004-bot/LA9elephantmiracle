"""공용 피처 생성 모듈 — 학습(train.py)과 추론(submit/script.py)에서 동일하게 사용한다.

build_features()가 train/추론 양쪽에서 호출되는 단일 진입점이다. 이렇게 묶어두면
학습과 추론의 피처 생성 로직이 갈라지는 실수(가장 흔한 제출 오류 원인)를 막을 수 있다.

Phase 2에서 결정된 기본값(DEFAULT_*)을 그대로 쓰면 최종 파이프라인이 되고,
실험 스크립트(notebooks/phase2_experiment.py)에서는 옵션을 바꿔가며 비교한다.
"""
import numpy as np
import pandas as pd

ID = "row_id"
TARGET = "control_success"
RAW_CAT_COLS = ["top_bottom", "game_type", "base_state"]

# n 컬럼과 함께 cold-start smoothing할 rate 컬럼 그룹
SMOOTH_GROUPS = {
    "asof_pitcher_n": [
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
    ],
    "asof_batter_n": [
        "asof_batter_success_rate", "asof_batter_middle_rate",
    ],
    "asof_pitcher_pitchmix_n": [
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ],
}

# 표본 수(n) 컬럼이 없어 smoothing 공식을 못 쓰는 rate 컬럼 — 결측 유지 vs 리그평균 대체 비교 대상
NO_N_RATE_COLS = [
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
]

# Phase 2 실험(phase2_features.md)으로 확정된 기본값
DEFAULT_SMOOTH_K = 200
DEFAULT_FILL_NO_N = True


def get_raw_feature_list(test_csv_path="./data/test.csv"):
    """test.csv 헤더 기준으로 사용할 원본 피처 목록을 정한다."""
    test_cols = pd.read_csv(test_csv_path, encoding="utf-8-sig", nrows=0).columns
    return [c for c in test_cols if c != ID]


def fit_stats(train_df):
    """학습 구간(season<=2023)에서만 계산해야 하는 통계량.

    리그 평균, li 분위수 경계 등 — 검증/테스트 데이터로 계산하면 정보 누수이므로
    반드시 학습 데이터로만 호출한다.
    """
    stats = {}
    for rate_cols in SMOOTH_GROUPS.values():
        for col in rate_cols:
            stats[f"league_avg__{col}"] = float(train_df[col].mean())
    for col in NO_N_RATE_COLS:
        stats[f"league_avg__{col}"] = float(train_df[col].mean())

    li_edges = train_df["li"].quantile([0, .25, .5, .75, 1.0]).tolist()
    li_edges[0], li_edges[-1] = -np.inf, np.inf
    stats["li_bin_edges"] = li_edges
    return stats


def add_derived_features(
    df, stats,
    include_count=True,
    include_matchup=True,
    include_situational=True,
    include_smoothing=True,
    include_interactions=True,
    smoothing_k=DEFAULT_SMOOTH_K,
    fill_no_n=DEFAULT_FILL_NO_N,
):
    """원본 컬럼으로 파생 피처를 추가한다. 옵션으로 그룹별 on/off, k 비교가 가능하다."""
    df = df.copy()

    if include_count:
        df["count_state"] = df["balls_before"].astype(str) + df["strikes_before"].astype(str)
        df["is_two_strikes"] = (df["strikes_before"] == 2).astype(int)
        df["is_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(int)
        df["is_pitcher_favor"] = (df["strikes_before"] > df["balls_before"]).astype(int)
        df["is_batter_favor"] = (df["balls_before"] > df["strikes_before"]).astype(int)

    if include_matchup:
        df["hand_matchup"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)
        df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)

    if include_situational:
        df["inning_group"] = pd.cut(
            df["inning"], bins=[0, 3, 6, 99], labels=["1-3", "4-6", "7+"]
        ).astype(str)
        df["scoring_position"] = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype(int)
        df["score_diff_abs"] = df["score_diff_pitcher_team"].abs()
        df["li_bucket"] = pd.cut(df["li"], bins=stats["li_bin_edges"], labels=False).astype(float)

    if include_smoothing:
        for n_col, rate_cols in SMOOTH_GROUPS.items():
            n = df[n_col].fillna(0)
            for col in rate_cols:
                league_avg = stats[f"league_avg__{col}"]
                smoothed = (df[col].fillna(0) * n + smoothing_k * league_avg) / (n + smoothing_k)
                df[f"smooth_{col}"] = smoothed
        for col in NO_N_RATE_COLS:
            if fill_no_n:
                df[col] = df[col].fillna(stats[f"league_avg__{col}"])
            # else: NaN 그대로 두고 LightGBM이 분기에서 자동 처리

    # reverse_x_batter_favor: 3구간x3시드 검증에서는 개선(0.248104→0.248088)이었으나
    # 2026-08-09 실제 리더보드 제출에서 748.41→744.37로 악화되어 되돌림(리더보드가 진짜
    # 검증이라는 이 프로젝트의 원칙 유지). include_interactions 인자는 향후 재검증 대비 유지.
    if include_interactions:
        pass

    return df


def get_categorical_columns(include_count=True, include_matchup=True, include_situational=True):
    cats = list(RAW_CAT_COLS)
    if include_count:
        cats.append("count_state")
    if include_matchup:
        cats.append("hand_matchup")
    if include_situational:
        cats.append("inning_group")
    return cats


def get_category_levels(df, cat_cols):
    """카테고리형 컬럼별 고정 값 목록을 만든다.

    LightGBM은 pandas category dtype을 내부적으로 정수 코드로 인코딩해서 쓴다.
    학습 때와 추론 때 각자 astype("category")를 따로 하면, 그 데이터에 등장한
    값의 집합이 다를 경우 같은 값이라도 다른 코드로 인코딩되어 예측이 틀어질 수
    있다(특히 5행짜리 배포용 샘플 test.csv처럼 값이 몇 개 안 나오는 경우). 그래서
    학습 데이터 기준으로 값 목록을 고정해두고 추론 시에도 그대로 재사용한다.
    """
    return {c: sorted(df[c].astype(str).unique().tolist()) for c in cat_cols}


def apply_category_levels(df, category_levels):
    """get_category_levels()로 저장해둔 고정 값 목록을 그대로 적용한다."""
    df = df.copy()
    for col, levels in category_levels.items():
        df[col] = pd.Categorical(df[col].astype(str), categories=levels)
    return df


def build_features(df, raw_features, stats=None, **kwargs):
    """원본 입력 df에서 모델 입력 X를 만든다. train/추론 공용 함수.

    stats가 None이면 파생 피처 없이 원본 컬럼만 선택한다(Phase 1 호환).
    """
    out = df[raw_features].copy()
    if stats is not None:
        out = add_derived_features(out, stats, **kwargs)
    return out
