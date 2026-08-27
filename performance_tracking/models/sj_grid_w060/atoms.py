# -*- coding: utf-8 -*-
"""sj_final [축1] 전처리 원자 — sj `v85_preprocess_screen` 의 원자를 **168 위로 이식**한다.

WORKFLOW_COMPARISONS.md §30.2(b) 가 지목한 축이다. 원본은 sj 279 프레임(pandas)에서
돌지만, 여기서는 **X168 자체가 원시값을 담고 있어서** 프레임을 다시 만들 필요가 없다
(0~43번 열이 train.csv 원시 컬럼 그대로다). 행 정렬이 자동으로 보장된다.

§30.2 표의 중복 판정 두 개를 고쳤다.
  · `count_multiscale` 을 "중복(dom_count 27열)" 이라 했는데, cw168 의 `cnt_*` 27열은
    **볼카운트 상태**(0-0/0-1/…)이지 표본수가 아니다. `asof_*_n` 을 다루는 이 원자와
    겹치지 않는다. 되살린다.
  · `trackman_quality` 는 "중복(T2 55열)" 이라 했는데, 그 55열은 **물리량 자체**이고
    이 원자는 **결측·스타일·분산 요약**이다. 겹치지 않는다. 다만 원본이 `tm500_*`
    전용이라 cw 의 `tm_{fa,br,of}_*` 구조로 이식했다.

★ preprocess_lab/RESULTS.md 의 1번 교훈 — **단독으로 거르면 안 된다.**
  `trackman_quality` 는 단독 −3.11 인데 최상위 조합 다섯 개 전부에 들어 있었다.
  그래서 판정은 단독 스크리닝이 아니라 **조합 탐색**으로 한다 (`run_atoms.py --beam`).

모든 통계는 `tr`(학습 마스크) 안에서만 적합한다 — 시간 인과 규칙.
"""

from __future__ import annotations

import warnings

import numpy as np

# ── 168 안의 원시 컬럼 이름 (meta.json 의 names 와 일치) ──────────────────────
ID_COLS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]
COUNT_COLS = ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]
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

EPS = 1e-6


class Frame:
    """이름으로 X168 의 열을 꺼내는 얇은 래퍼. 없는 열은 즉시 실패한다."""

    def __init__(self, X, names, tr):
        self.X = X
        self.idx = {c: i for i, c in enumerate(names)}
        self.tr = tr
        self.n = X.shape[0]

    def has(self, c):
        return c in self.idx

    def col(self, c):
        if c not in self.idx:
            raise KeyError(f"168 에 없는 열: {c}")
        return np.asarray(self.X[:, self.idx[c]], dtype=np.float64)

    def cols(self, cs):
        return np.column_stack([self.col(c) for c in cs])

    def startswith(self, *pre):
        return [c for c in self.idx if c.startswith(pre)]


def _logit(v):
    v = np.clip(v, EPS, 1 - EPS)
    return np.log(v / (1 - v))


def _robust_z(F, columns):
    """학습행의 중앙값·IQR 로 표준화. 원본 v85.robust_z 와 같은 식."""
    out = []
    for c in columns:
        v = F.col(c)
        t = v[F.tr]
        t = t[np.isfinite(t)]
        if t.size == 0:
            out.append(np.zeros_like(v))
            continue
        med = float(np.median(t))
        q1, q3 = np.percentile(t, [25, 75])
        scale = float(q3 - q1) or 1.0
        out.append((v - med) / scale)
    return np.column_stack(out)


# --------------------------------------------------------------------------- #
# 원자들 — 전부 (Frame, fold) -> {이름: 1차원 배열}
# --------------------------------------------------------------------------- #
def id_freq(F, fold):
    """원본 ID 는 남기고 빈도 인코딩을 **더한다**.

    sj 원본은 ID 열을 제거하고 빈도로 갈아끼웠다(−57 이 나온 drop_ids 와 대비된다).
    여기서는 cw168 이 ID 를 이미 EB 인코딩(enc_*, season_*)으로도 쓰고 있으므로
    제거하지 않고 더하기만 한다. 제거 여부 자체는 별도 arm 으로 잰다.
    """
    e = {}
    for c in ID_COLS:
        v = F.col(c)
        u, cnt = np.unique(v[F.tr], return_counts=True)
        pos = np.searchsorted(u, v)
        pos = np.clip(pos, 0, len(u) - 1)
        hit = u[pos] == v
        freq = np.where(hit, cnt[pos], 0.0)
        e[f"pa_{c}_logfreq"] = np.log1p(freq)
        e[f"pa_{c}_unseen"] = (freq == 0).astype(np.float64)
    return e


def temporal(F, fold):
    month, day = F.col("game_month"), F.col("game_dayofweek")
    inning, season = F.col("inning"), F.col("season")
    return {
        "pa_month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
        "pa_month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
        "pa_day_sin": np.sin(2 * np.pi * day / 7.0),
        "pa_day_cos": np.cos(2 * np.pi * day / 7.0),
        "pa_inning_clipped": np.minimum(inning, 10.0),
        "pa_inning_extra": np.maximum(inning - 9.0, 0.0),
        "pa_years_to_pred": season - fold,
        "pa_season_month_progress": (season - fold) * 12.0 + month,
    }


def context(F, fold):
    e = {}
    for c in ("run_top_before", "run_bot_before", "run_total_before",
              "score_diff_home", "score_diff_pitcher_team"):
        v = F.col(c)
        e[f"pa_signed_log_{c}"] = np.sign(v) * np.log1p(np.abs(v))
    li = np.clip(F.col("li"), 0, None)
    e["pa_log1p_li"] = np.log1p(li)
    e["pa_li_capped_3"] = np.minimum(li, 3.0)
    e["pa_expectancy_centered"] = F.col("home_win_expectancy") - 0.5
    e["pa_runner_pressure"] = (F.col("runner_on_1b") + 2.0 * F.col("runner_on_2b")
                               + 3.0 * F.col("runner_on_3b"))
    e["pa_outs_remaining"] = 3.0 - F.col("outs_before")
    return e


def rate_ms(F, fold):
    """성공률을 여러 강도로 EB 평활 + reliability. 사전확률은 학습행 중앙값."""
    e = {}
    for name, rc, cc in RATE_SPECS:
        rate = F.col(rc)
        cnt = np.nan_to_num(F.col(cc), nan=0.0)
        t = rate[F.tr]
        prior = float(np.nanmedian(t)) if np.isfinite(t).any() else 0.5
        filled = np.where(np.isfinite(rate), rate, prior)
        for s in (50.0, 500.0, 1000.0):
            e[f"pa_{name}_sm{int(s)}"] = (cnt * filled + s * prior) / (cnt + s)
            e[f"pa_{name}_rel{int(s)}"] = cnt / (cnt + s)
    return e


def rate_geom(F, fold):
    """logit 변환 + 투타 격차 + 구질 엔트로피/로그비.

    cw168 이 이미 `pitchmix_entropy`(64) 와 `batter_minus_pitcher`(65) 를 갖고 있으나
    **logit 변환 자체는 없다.** 겹치는 두 열은 빼고 나머지만 만든다.
    """
    e = {}
    for _, rc, _ in RATE_SPECS:
        e[f"pa_logit_{rc}"] = _logit(F.col(rc))
    for c in ("home_win_expectancy", "away_win_expectancy"):
        e[f"pa_logit_{c}"] = _logit(F.col(c))
    e["pa_success_logit_gap"] = (_logit(F.col("asof_pitcher_success_rate"))
                                 - _logit(F.col("asof_batter_success_rate")))
    e["pa_expectancy_logit_gap"] = (_logit(F.col("home_win_expectancy"))
                                    - _logit(F.col("away_win_expectancy")))
    mix = np.clip(F.cols(["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
                          "asof_pitcher_offspeed_rate"]), EPS, 1.0)
    s = np.nansum(mix, axis=1, keepdims=True)
    nz = mix / np.where(s > 0, s, np.nan)
    e["pa_pitchmix_concentration"] = np.nansum(nz ** 2, axis=1)
    e["pa_fb_br_logratio"] = np.log(mix[:, 0] / mix[:, 1])
    e["pa_fb_of_logratio"] = np.log(mix[:, 0] / mix[:, 2])
    return e


def count_ms(F, fold):
    """표본수의 sqrt · 버킷 · reliability. cw168 은 log_pitcher_n/log_batter_n 뿐이다."""
    e = {}
    bins = np.array([0, 25, 100, 500, 1000, 2000, 4000], np.float64)
    for c in COUNT_COLS:
        v = np.nan_to_num(F.col(c), nan=0.0)
        e[f"pa_sqrt_{c}"] = np.sqrt(np.clip(v, 0, None))
        e[f"pa_bucket_{c}"] = np.searchsorted(bins, v, side="right").astype(np.float64)
        for s in (25.0, 100.0, 500.0, 2000.0):
            e[f"pa_{c}_rel{int(s)}"] = v / (v + s)
    return e


def recent(F, fold):
    """최근 등판의 가중합·기울기·곡률·shock. cw168 은 prev1/3/5 원시값과 delta 뿐이다."""
    e = {}
    w = np.array([5.0, 3.0, 1.0])
    for tag, cols, career in (("success", PREV_SUCCESS, "asof_pitcher_success_rate"),
                              ("middle", PREV_MIDDLE, "asof_pitcher_middle_rate")):
        V = F.cols(cols)
        ok = np.isfinite(V)
        den = (ok * w).sum(axis=1)
        e[f"pa_recent_{tag}_weighted"] = np.where(
            den > 0, np.nansum(np.nan_to_num(V) * w, axis=1) / np.where(den > 0, den, 1), np.nan)
        cv = F.col(career)
        e[f"pa_recent_{tag}_slope"] = V[:, 0] - V[:, 2]
        e[f"pa_recent_{tag}_curv"] = V[:, 0] - 2.0 * V[:, 1] + V[:, 2]
        e[f"pa_recent_{tag}_shock"] = V[:, 0] - cv
        e[f"pa_recent_{tag}_abs_shock"] = np.abs(V[:, 0] - cv)
        e[f"pa_recent_{tag}_missing"] = (~ok).sum(axis=1).astype(np.float64)
    return e


def tm_quality(F, fold):
    """TrackMan 결측·스타일·분산 요약. **cw 의 tm_{fa,br,of}_* 구조로 이식했다.**

    원본은 `tm500_latest_*_mean/_std` 와 `*_minus_recent` 를 봤다. cw168 의 대응은
    `tm_{fa,br,of}_*_mean` / `_std` 이고, "이동" 자리에는 구종 간 차분 `tm_d_*` 를 쓴다.
    표본수 자리(`tm500_total_pitches`)는 `tm_{fa,br,of}_n` 합으로 대신한다.
    """
    tm = sorted(F.startswith("tm_"))
    means = [c for c in tm if c.endswith("_mean")]
    stds = [c for c in tm if c.endswith("_std")]
    diffs = [c for c in tm if c.startswith("tm_d_")]
    ns = [c for c in ("tm_fa_n", "tm_br_n", "tm_of_n") if F.has(c)]
    R = F.cols(tm)
    e = {"pa_tm_missing_count": np.isnan(R).sum(axis=1).astype(np.float64),
         "pa_tm_missing_ratio": np.isnan(R).mean(axis=1)}
    # TrackMan 프로필이 아예 없는 행(≈20%)은 전열 NaN 이라 nanmean 이 빈 슬라이스
    # 경고를 낸다. 결과 NaN 은 CatBoost 가 그대로 받는 정상값이므로 경고만 막는다.
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if means:
            Z = _robust_z(F, means)
            e["pa_tm_style_l2"] = np.sqrt(np.nanmean(Z ** 2, axis=1))
            e["pa_tm_style_mean"] = np.nanmean(Z, axis=1)
            e["pa_tm_style_std"] = np.nanstd(Z, axis=1)
        if stds:
            Z = _robust_z(F, stds)
            e["pa_tm_disp_mean"] = np.nanmean(Z, axis=1)
            e["pa_tm_disp_l2"] = np.sqrt(np.nanmean(Z ** 2, axis=1))
        if diffs:
            Z = _robust_z(F, diffs)
            e["pa_tm_shift_l2"] = np.sqrt(np.nanmean(Z ** 2, axis=1))
            e["pa_tm_shift_mean"] = np.nanmean(Z, axis=1)
        if ns:
            N = F.cols(ns)
            tot = np.nansum(N, axis=1)
            e["pa_tm_log_total"] = np.log1p(np.clip(tot, 0, None))
            fa = np.nan_to_num(F.col("tm_fa_n"), nan=0.0)
            e["pa_tm_fa_share"] = fa / np.maximum(tot, 1.0)
    return e


ATOMS = {
    "id_freq": id_freq,
    "temporal": temporal,
    "context": context,
    "rate_ms": rate_ms,
    "rate_geom": rate_geom,
    "count_ms": count_ms,
    "recent": recent,
    "tm_quality": tm_quality,
}

# §30.2 가 투입 판정한 것 (○/△). 나머지 넷은 랩 결과가 좋아서 되살린 것들이다.
PLANNED = ["id_freq", "temporal", "context", "rate_ms"]
REVIVED = ["rate_geom", "count_ms", "recent", "tm_quality"]


def build(X, names, tr, fold, atom_names):
    """원자 여러 개를 합쳐 (E, enames) 반환. 이름 충돌은 즉시 실패."""
    F = Frame(X, names, tr)
    cols, enames = [], []
    for a in atom_names:
        if a not in ATOMS:
            raise KeyError(f"모르는 원자: {a}  (가능: {sorted(ATOMS)})")
        for k, v in ATOMS[a](F, fold).items():
            if k in enames:
                raise ValueError(f"열 이름 충돌: {k}")
            enames.append(k)
            cols.append(np.asarray(v, np.float32))
    if not cols:
        return np.zeros((X.shape[0], 0), np.float32), []
    return np.column_stack(cols).astype(np.float32), enames
