# -*- coding: utf-8 -*-
"""시즌 폼 피처 — 학습과 추론이 **같은 함수**를 쓴다.

이 프로젝트의 실패 세 건(v6 852, 스플릿 -52%, 검증 오측)이 전부
"학습 때와 추론 때 피처가 다른 규칙으로 만들어진" 문제였다.
그래서 계산의 핵심을 함수 하나(features)로 고정하고, 양쪽이 그것만 호출한다.

원리:
    asof_pitcher_n 은 시즌을 넘어 통산으로 쌓인다 (2019 0→2757, 2020 2758→5427 ...).
    통산 3,000구 투수가 올해 500구를 던져도 asof_rate 는 거의 안 움직인다.
    그래서 "올 시즌 어떤가" 를 모델이 못 본다.

        그 행에서       통산성공수_지금  = asof_n × asof_rate
        룩업에서        통산성공수_전년말 = lut[pitcher_id]
        ─────────────────────────────────────────────
        올 시즌 성적 = 두 값의 차이

    prev1/3/5경기(수십 구)와 통산(수천 구) 사이의 빈 시간 척도를 메운다.

규정 준수 (데이콘 Phase 2 공지):
    쓰는 것은 (1) 그 행의 asof_n · asof_rate · pitcher_id/batter_id
             (2) 학습 데이터로만 만든 룩업 테이블
    평가 데이터의 다른 행을 보지 않는다. test.csv 에 1행만 있어도 결과가 같다.

누수 방지:
    asof_* 는 그 투구 **직전**까지의 값이다 (성공수 증분 = 그 행의 결과임을 실측 확인).
    따라서 시즌 성적에 그 행 자신의 결과가 들어가지 않는다.
    학습 시에는 s년 행에 s-1년 말 룩업을, 추론 시에는 2024년 말 룩업을 쓴다. 규칙이 같다.

검증 결과 (val 2024 / 2022):
    기준 693.7 / 2309.6  →  +시즌폼 748.6 / 2418.6   (+7.93% / +4.72%)
    플라시보(투수 대응 섞음)  682.9 / 2309.5           (-1.55% / -0.00%)  ← 누수 아님
"""

import numpy as np

SHRINK_K = 150.0
N_FEATURES = 8          # 투수 4 + 타자 4
NAMES = ["season_log_n_p", "season_rate_p", "season_rate_shr_p", "season_delta_p",
         "season_log_n_b", "season_rate_b", "season_rate_shr_b", "season_delta_b"]


# ───────────────── 핵심 계산 (학습·추론 공용) ─────────────────

def features(asof_n, asof_rate, n0, s0, prior):
    """통산 상태와 전년말 상태로부터 시즌 폼 4피처.

    asof_n, asof_rate : 그 행의 값 (그 투구 직전까지의 통산)
    n0, s0            : 전년말 통산 (투구수, 성공수) — 룩업에서
    prior             : 결측 대비 사전값 (학습 데이터 전체 성공률)
    """
    n = np.nan_to_num(np.asarray(asof_n, dtype=np.float64), nan=0.0)
    rate = np.asarray(asof_rate, dtype=np.float64)
    cum = n * np.nan_to_num(rate, nan=0.0)

    sn = np.maximum(n - n0, 0.0)                       # 올 시즌 투구수
    ss = np.clip(cum - s0, 0.0, None)                  # 올 시즌 성공수
    career = np.where(np.isnan(rate), prior, np.nan_to_num(rate, nan=prior))
    shr = (ss + SHRINK_K * career) / (sn + SHRINK_K)   # 통산 쪽으로 축소
    raw = np.where(sn > 0, ss / np.maximum(sn, 1.0), career)
    return np.column_stack([np.log1p(sn), raw, shr, shr - career]).astype(np.float32)


def lookup(ids, keys, vals):
    """id → (n0, s0). 없으면 (0, 0) — 신인은 통산 전체가 곧 올 시즌이다."""
    ids = np.asarray(ids)
    if len(keys) == 0:
        return np.zeros(len(ids)), np.zeros(len(ids))
    pos = np.clip(np.searchsorted(keys, ids), 0, len(keys) - 1)
    hit = keys[pos] == ids
    return np.where(hit, vals[pos, 0], 0.0), np.where(hit, vals[pos, 1], 0.0)


def apply_all(df, lut, prior):
    """DataFrame → 8피처. 추론에서 이 함수 하나만 부르면 된다."""
    out = []
    for side, idc, nc, rc in (("p", "pitcher_id", "asof_pitcher_n",
                               "asof_pitcher_success_rate"),
                              ("b", "batter_id", "asof_batter_n",
                               "asof_batter_success_rate")):
        n0, s0 = lookup(df[idc].to_numpy(np.int64),
                        lut["%s_key" % side], lut["%s_val" % side])
        out.append(features(df[nc].to_numpy(dtype=np.float64),
                            df[rc].to_numpy(dtype=np.float64), n0, s0, prior))
    return np.concatenate(out, axis=1)


# ───────────────── 룩업 생성 (학습 데이터에서만) ─────────────────

def end_state(df, upto, idcol, ncol, ratecol, ycol="control_success"):
    """upto 시즌 말 기준 id → (통산 n, 통산 성공수)."""
    h = df[df.season <= upto]
    if len(h) == 0:
        return np.array([], dtype=np.int64), np.zeros((0, 2))
    last = h.sort_values(ncol).groupby(idcol, sort=False).tail(1)
    n = last[ncol].to_numpy(dtype=np.float64)
    s = n * np.nan_to_num(last[ratecol].to_numpy(dtype=np.float64), nan=0.0)
    # 그 마지막 투구의 결과까지 더해야 '시즌 말' 상태가 된다
    s = s + last[ycol].to_numpy(dtype=np.float64)
    n = n + 1.0
    k = last[idcol].to_numpy(np.int64)
    o = np.argsort(k)
    return k[o], np.column_stack([n, s])[o]


def build_lookup(df, upto):
    """추론용 룩업 하나. upto=2024 이면 2025 예측에 쓴다."""
    pk, pv = end_state(df, upto, "pitcher_id", "asof_pitcher_n",
                       "asof_pitcher_success_rate")
    bk, bv = end_state(df, upto, "batter_id", "asof_batter_n",
                       "asof_batter_success_rate")
    return {"p_key": pk, "p_val": pv, "b_key": bk, "b_val": bv,
            "prior": np.array([float(df.control_success.mean())])}


def build_training_features(df):
    """학습용 8피처. s년 행에는 s-1년 말 룩업을 쓴다 (시즌별 확장).

    추론과 규칙이 같다: 어떤 행도 자기 시즌의 다른 행을 보지 않는다.
    """
    prior = float(df.control_success.mean())
    sea = df.season.to_numpy()
    out = np.zeros((len(df), N_FEATURES), dtype=np.float32)
    for s in sorted(set(sea.tolist())):
        m = sea == s
        lut = build_lookup(df, s - 1)
        out[m] = apply_all(df.loc[m], lut, prior)
    return out
