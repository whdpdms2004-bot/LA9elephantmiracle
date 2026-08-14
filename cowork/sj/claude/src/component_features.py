"""submit_024 성분 모델 전용 피처 빌더 — 학습과 추론이 같은 코드를 쓴다.

script.py 는 train.csv 없이 실행되므로 필요한 상수를 전부 spec(JSON)에 담는다.
    priors        stateless 파생용 prior 5개
    rate_median   prof200 수축용 rate 중앙값 8개
    prev_median   최근폼 결측 대치용 중앙값 6개
    cat_map       top_bottom / game_type / base_state 명시적 매핑 (cat.codes 의존 금지)
    columns       피처 순서 (XGBoost 는 순서까지 고정해야 한다)
    platoon       (pitcher_id, batter_hand) -> split, rel  룩업

행 단위로만 계산한다. test 의 다른 행을 참조하지 않으므로
predict(단독 행) == predict(전체)[i] 가 성립한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "control_success"

RATES = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
         "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
         "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
         "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
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
STRENGTH = 200.0
LAM_PROF = 200.0
PLATOON_K = 300


def make_spec(train: pd.DataFrame) -> dict:
    """학습 데이터에서 추론에 필요한 상수를 전부 뽑는다."""
    spec = {
        "priors": {
            "pitcher_success": float(train[TARGET].mean()),
            "pitcher_reverse": float(train["asof_pitcher_reverse_rate"].median()),
            "pitcher_middle": float(train["asof_pitcher_middle_rate"].median()),
            "batter_success": float(train[TARGET].mean()),
            "batter_middle": float(train["asof_batter_middle_rate"].median()),
        },
        "rate_median": {c: float(train[c].median()) for c in RATES},
        "prev_median": {c: float(train[c].median()) for c in PREV_S + PREV_M},
        "cat_map": {c: {str(v): i for i, v in
                        enumerate(sorted(train[c].astype(str).unique()))}
                    for c in CAT_COLS},
        "strength": STRENGTH,
        "lam_prof": LAM_PROF,
        "platoon_k": PLATOON_K,
    }
    return spec


def make_platoon_table(train: pd.DataFrame, K: int = PLATOON_K) -> pd.DataFrame:
    """split(p,h) = EB(투수 p, 타자손 h) - EB(투수 p 전체).

    주효과를 빼는 것이 핵심 — 안 빼면 asof_pitcher_success_rate 와 중복된다
    (찬우 실험: 투수 주효과 단독 기여 정확히 0.0).
    """
    d = train[["pitcher_id", "batter_hand", TARGET]].rename(
        columns={TARGET: "y"})
    league = float(d["y"].mean())
    g_all = d.groupby("pitcher_id")["y"].agg(["sum", "size"])
    eb_all = (g_all["sum"] + K * league) / (g_all["size"] + K)
    g_ph = d.groupby(["pitcher_id", "batter_hand"])["y"].agg(["sum", "size"])
    eb_ph = (g_ph["sum"] + K * league) / (g_ph["size"] + K)
    out = pd.DataFrame({
        "platoon_split": eb_ph - eb_ph.index.get_level_values(0).map(eb_all),
        "platoon_rel": g_ph["size"] / (g_ph["size"] + K),
    }).reset_index()
    out.attrs["league_mean"] = league
    return out


def build(frame: pd.DataFrame, spec: dict, platoon: pd.DataFrame) -> pd.DataFrame:
    """행 단위 피처 생성. frame 은 train/test 어느 쪽이든 입력 48컬럼 구조."""
    pri = spec["priors"]
    st = float(spec["strength"])
    lam = float(spec["lam_prof"])
    x = frame
    out = {}

    # --- 원본 47 (row_id, target 제외). 범주는 명시 매핑
    for c in x.columns:
        if c in ("row_id", TARGET):
            continue
        if c in CAT_COLS:
            out[c] = x[c].astype(str).map(spec["cat_map"][c]).fillna(-1).to_numpy(np.float64)
        else:
            out[c] = pd.to_numeric(x[c], errors="coerce").to_numpy(np.float64)

    b = out["balls_before"]; s = out["strikes_before"]
    ph = out["pitcher_hand"]; bh = out["batter_hand"]
    n = out["asof_pitcher_n"]
    out["count_state"] = b * 3 + s
    out["handedness_matchup"] = ph * 2 + bh
    out["runner_out_state"] = out["num_runners_on"] * 3 + out["outs_before"]
    out["score_abs"] = np.abs(out["score_diff_pitcher_team"])
    out["late_inning"] = (out["inning"] >= 7).astype(np.float64)
    out["high_leverage"] = (out["li"] >= 2).astype(np.float64)
    out["log1p_asof_pitcher_n"] = np.log1p(n)
    out["log1p_asof_batter_n"] = np.log1p(out["asof_batter_n"])
    for k in (1, 3, 5):
        out[f"pitcher_success_delta_prev{k}"] = (
            out[f"asof_pitcher_prev{k}_game_success_rate"]
            - out["asof_pitcher_success_rate"])
        out[f"pitcher_middle_delta_prev{k}"] = (
            out[f"asof_pitcher_prev{k}_game_middle_rate"]
            - out["asof_pitcher_middle_rate"])
    out["ball_strike_gap"] = (out["asof_pitcher_ball_rate"]
                              - out["asof_pitcher_strike_rate"])
    for name, rate_col, n_col in RATE_SPECS:
        nn = out[n_col]
        rate = np.where(np.isnan(out[rate_col]), pri[name], out[rate_col])
        out[f"{name}_is_missing"] = np.isnan(out[rate_col]).astype(np.float64)
        out[f"{name}_smoothed"] = (nn * rate + st * pri[name]) / (nn + st)
        out[f"{name}_reliability"] = nn / (nn + st)

    # --- 수축 프로파일
    for c in RATES:
        med = spec["rate_median"][c]
        r = np.where(np.isnan(out[c]), med, out[c])
        out[f"prof200_{c}"] = (n * r + lam * med) / (n + lam)

    # --- 최근 폼 파생
    ps = {c: np.where(np.isnan(out[c]), spec["prev_median"][c], out[c])
          for c in PREV_S + PREV_M}
    out["prev_trend_s"] = ps[PREV_S[0]] - ps[PREV_S[2]]
    out["prev_trend_m"] = ps[PREV_M[0]] - ps[PREV_M[2]]
    out["prev_std_s"] = np.std(np.vstack([ps[c] for c in PREV_S]), axis=0)
    out["prev_std_m"] = np.std(np.vstack([ps[c] for c in PREV_M]), axis=0)
    out["prev_miss_cnt"] = sum(np.isnan(out[c]).astype(np.float64)
                               for c in PREV_S + PREV_M)
    for k, (cs, cm) in enumerate(zip(PREV_S, PREV_M)):
        out[f"faildir_{k}"] = ps[cm] - (1 - ps[cs])
    out["rel200"] = n / (n + lam)

    # --- 플래툰 (학습 데이터로 만든 정적 룩업을 행 단위 조인)
    key = pd.MultiIndex.from_arrays(
        [pd.to_numeric(x["pitcher_id"]), pd.to_numeric(x["batter_hand"])])
    pt = platoon.set_index(["pitcher_id", "batter_hand"]).reindex(key)
    sp = pt["platoon_split"].fillna(0.0).to_numpy(np.float64)
    rel = pt["platoon_rel"].fillna(0.0).to_numpy(np.float64)
    out["platoon_split"] = sp
    out["platoon_rel"] = rel
    out["platoon_split_w"] = sp * rel

    return pd.DataFrame(out)


def matrix(frame: pd.DataFrame, spec: dict, platoon: pd.DataFrame) -> np.ndarray:
    """spec['columns'] 순서로 float32 행렬을 만든다."""
    d = build(frame, spec, platoon)
    return d[spec["columns"]].to_numpy(np.float32)
