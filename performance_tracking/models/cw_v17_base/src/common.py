# -*- coding: utf-8 -*-
"""학습/추론이 공유하는 피처 생성 로직.

script.py 는 이 파일의 내용을 그대로 인라인으로 포함한다(제출 zip 구조상 단일 파일 권장).
여기서는 학습 코드가 import 해서 쓰고, 추론에서는 동일 코드를 복사해 사용한다.
"""

import numpy as np

ID_COL = "row_id"
TARGET_COL = "control_success"

# 원-핫으로 처리할 저카디널리티 범주형 (sklearn 네이티브 categorical 을 쓰지 않는 이유는
# 평가 서버와 sklearn/numpy 버전이 달라 pickle 호환을 보장할 수 없기 때문. 트리를 순수
# numpy 배열로 내보내 추론하므로 범주형 bitset 분기를 피하려는 목적도 있다.)
CAT_LEVELS = {
    "top_bottom": ["B", "T"],
    "game_type": ["F", "R"],
    "base_state": ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"],
}

# train.csv / test.csv 에 공통으로 존재하는 원본 수치형 컬럼 순서
NUM_COLS = [
    "season", "game_month", "game_dayofweek", "inning",
    "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]


def derived_names():
    """파생 피처 이름. build_features 의 출력 순서와 반드시 일치해야 한다."""
    return [
        "count_state",          # balls*3 + strikes (0~11)
        "is_two_strike",
        "is_three_ball",
        "abs_score_diff",
        "pitcher_middle_x_n",   # middle_rate 를 표본수로 신뢰가중
        "pitcher_form_delta",   # 최근 3경기 성공률 - 통산 성공률 (폼)
        "pitcher_form_delta5",
        "pitcher_middle_delta",
        "pitchmix_entropy",
        "batter_minus_pitcher", # 타자 상대 성공률 - 투수 성공률
        "log_pitcher_n",
        "log_batter_n",
    ]


# ---------------------------------------------------------------------------
# 플래툰 스플릿 인코딩
#
# 잔차 분석 결과 앙상블이 남긴 신호 중 유일하게 연도를 넘어 재현되는 것이
# "투수 x 타자좌우", "타자 x 투수좌우" 상호작용이었다. 투수·타자의 주효과는
# asof_* 가 이미 잡고 있어 추가 이득이 0 이므로, 주효과를 뺀 **스플릿만** 쓴다.
#
# 누수 방지: 학습 행(시즌 S)은 S 미만 시즌으로만 인코딩을 만든다. 평가(2025)는
# 2019~2024 전체를 쓴다. 평가 데이터의 다른 행은 전혀 사용하지 않는다.
# ---------------------------------------------------------------------------

ENC_K = 100.0          # 경험적 베이즈 축소 강도 (val 2024: K=300 619.3, K=100 627.8, K=50 595.6)
ENC_NAMES = ["enc_platoon_split", "enc_batter_split",
             "enc_platoon_n", "enc_batter_split_n"]


def _eb(keys, y, k, prior):
    u, inv = np.unique(keys, return_inverse=True)
    s = np.bincount(inv, weights=y)
    n = np.bincount(inv).astype(np.float64)
    return u, (s + k * prior) / (n + k), n


def build_encodings(pitcher_id, batter_id, pitcher_hand, batter_hand, y):
    """과거 시즌 투구로부터 스플릿 인코딩 테이블을 만든다."""
    y = np.asarray(y, dtype=np.float64)
    prior = float(y.mean())
    pk = pitcher_id.astype(np.int64) * 10 + batter_hand.astype(np.int64)
    bk = batter_id.astype(np.int64) * 10 + pitcher_hand.astype(np.int64)
    pu, pe, pn = _eb(pk, y, ENC_K, prior)
    pau, pae, _ = _eb(pitcher_id.astype(np.int64), y, ENC_K, prior)
    bu, be, bn = _eb(bk, y, ENC_K, prior)
    bau, bae, _ = _eb(batter_id.astype(np.int64), y, ENC_K, prior)
    # 스플릿 = (좌우별 성공률) − (전체 성공률)
    pmain = dict(zip(pau, pae))
    bmain = dict(zip(bau, bae))
    psplit = pe - np.array([pmain[k // 10] for k in pu])
    bsplit = be - np.array([bmain[k // 10] for k in bu])
    return dict(p_key=pu, p_split=psplit, p_n=pn,
                b_key=bu, b_split=bsplit, b_n=bn, prior=np.array([prior]))


def _lookup(keys, table_keys, table_vals, default=np.nan):
    idx = np.searchsorted(table_keys, keys)
    idx = np.clip(idx, 0, len(table_keys) - 1)
    ok = table_keys[idx] == keys
    out = np.full(len(keys), default, dtype=np.float64)
    out[ok] = table_vals[idx[ok]]
    return out


def encode_rows(pitcher_id, batter_id, pitcher_hand, batter_hand, enc):
    """인코딩 테이블을 행에 붙인다. 이력이 없으면 스플릿은 NaN, 표본수는 0."""
    pk = pitcher_id.astype(np.int64) * 10 + batter_hand.astype(np.int64)
    bk = batter_id.astype(np.int64) * 10 + pitcher_hand.astype(np.int64)
    ps = _lookup(pk, enc["p_key"], enc["p_split"])
    bs = _lookup(bk, enc["b_key"], enc["b_split"])
    pn = _lookup(pk, enc["p_key"], enc["p_n"], default=0.0)
    bn = _lookup(bk, enc["b_key"], enc["b_n"], default=0.0)
    return np.stack([ps, bs, np.log1p(pn), np.log1p(bn)], axis=1).astype(np.float32)


def _safe(a):
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def build_features(df):
    """DataFrame -> (X: float32 2D array, feature_names: list[str])

    train / test 어느 쪽에서 불러도 동일한 컬럼 순서를 만든다.
    """
    n = len(df)
    blocks, names = [], []

    num = np.empty((n, len(NUM_COLS)), dtype=np.float32)
    for i, c in enumerate(NUM_COLS):
        num[:, i] = df[c].to_numpy(dtype=np.float32, na_value=np.nan) \
            if hasattr(df[c], "to_numpy") else np.asarray(df[c], dtype=np.float32)
    blocks.append(num)
    names += NUM_COLS

    # 원-핫 (미지의 범주는 전부 0 -> 자연스럽게 "그 외" 취급)
    for c, levels in CAT_LEVELS.items():
        v = df[c].astype(str).to_numpy()
        oh = np.zeros((n, len(levels)), dtype=np.float32)
        for j, lv in enumerate(levels):
            oh[:, j] = (v == lv)
        blocks.append(oh)
        names += [f"{c}={lv}" for lv in levels]

    g = {c: num[:, NUM_COLS.index(c)] for c in NUM_COLS}
    balls, strikes = g["balls_before"], g["strikes_before"]
    pn = g["asof_pitcher_n"]
    bn = g["asof_batter_n"]
    fb, br, os_ = g["asof_pitcher_fastball_rate"], g["asof_pitcher_breaking_rate"], g["asof_pitcher_offspeed_rate"]
    mix = np.stack([_safe(fb), _safe(br), _safe(os_)], axis=1)
    mix = np.clip(mix, 1e-6, 1.0)
    ent = -(mix * np.log(mix)).sum(axis=1)

    der = np.stack([
        balls * 3.0 + strikes,
        (strikes >= 2).astype(np.float32),
        (balls >= 3).astype(np.float32),
        np.abs(g["score_diff_pitcher_team"]),
        g["asof_pitcher_middle_rate"] * (pn / (pn + 500.0)),
        g["asof_pitcher_prev3_game_success_rate"] - g["asof_pitcher_success_rate"],
        g["asof_pitcher_prev5_game_success_rate"] - g["asof_pitcher_success_rate"],
        g["asof_pitcher_prev3_game_middle_rate"] - g["asof_pitcher_middle_rate"],
        ent,
        g["asof_batter_success_rate"] - g["asof_pitcher_success_rate"],
        np.log1p(np.nan_to_num(pn, nan=0.0)),
        np.log1p(np.nan_to_num(bn, nan=0.0)),
    ], axis=1).astype(np.float32)
    blocks.append(der)
    names += derived_names()

    X = np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32)
    return X, names
