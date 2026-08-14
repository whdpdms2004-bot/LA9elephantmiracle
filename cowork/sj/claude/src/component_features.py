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


def _eb_split(actor: pd.Series, opp_hand: pd.Series, y: pd.Series,
              K: int) -> pd.DataFrame:
    """split(actor, opp_hand) = EB(actor x 상대손) - EB(actor 전체).

    주효과를 빼는 것이 핵심이다. 안 빼면 asof_*_success_rate 와 중복이고,
    정적 테이블에서는 자기 라벨이 그대로 샌다.
    실측: V1 에서 정적 '레벨'을 넣으니 direct_bss 705.7 -> 187.5 붕괴.
          V12 에서 타자 성분 '프로파일'도 748.4 -> 626.5 붕괴.
          '차이(split)'는 주효과가 상쇄돼 이 문제가 없다.
    """
    d = pd.DataFrame({"a": actor.to_numpy(), "h": opp_hand.to_numpy(),
                      "y": y.to_numpy()})
    league = float(d["y"].mean())
    g_a = d.groupby("a")["y"].agg(["sum", "size"])
    eb_a = (g_a["sum"] + K * league) / (g_a["size"] + K)
    g_ah = d.groupby(["a", "h"])["y"].agg(["sum", "size"])
    eb_ah = (g_ah["sum"] + K * league) / (g_ah["size"] + K)
    out = pd.DataFrame({
        "split": eb_ah - eb_ah.index.get_level_values(0).map(eb_a),
        "rel": g_ah["size"] / (g_ah["size"] + K),
    }).reset_index()
    out.attrs["league_mean"] = league
    return out


def make_platoon_table(train: pd.DataFrame, K: int = PLATOON_K) -> pd.DataFrame:
    """투수 플래툰. split(p,h) = EB(투수 p, 타자손 h) - EB(투수 p 전체)."""
    out = _eb_split(train["pitcher_id"], train["batter_hand"], train[TARGET], K)
    out = out.rename(columns={"a": "pitcher_id", "h": "batter_hand",
                              "split": "platoon_split", "rel": "platoon_rel"})
    out.attrs["league_mean"] = float(train[TARGET].mean())
    return out


def make_batter_platoon_table(train: pd.DataFrame, components: dict,
                              K: int = PLATOON_K) -> pd.DataFrame:
    """타자 플래툰 (V12 G4). 전역 + 성분별을 한 테이블로 낸다.

    투수 쪽은 전역 스플릿이 이미 있어서 성분별로 쪼개도 이득이 없었지만(V8 +0.16),
    타자 쪽은 전역조차 없던 상태라 성분별 정보가 그대로 새 정보다(V12 +1.22).

    components: {tag: 0/1 라벨 배열}. NaN 행은 해당 성분 집계에서 제외한다.
    """
    base = _eb_split(train["batter_id"], train["pitcher_hand"], train[TARGET], K)
    base = base.rename(columns={"a": "batter_id", "h": "pitcher_hand",
                                "split": "bat_platoon_split",
                                "rel": "bat_platoon_rel"})
    for tag, arr in components.items():
        m = ~np.isnan(arr)
        sub = _eb_split(train["batter_id"][m], train["pitcher_hand"][m],
                        pd.Series(arr[m]), K)
        sub = sub.rename(columns={"a": "batter_id", "h": "pitcher_hand",
                                  "split": f"bat_pl_{tag}"})[
            ["batter_id", "pitcher_hand", f"bat_pl_{tag}"]]
        base = base.merge(sub, on=["batter_id", "pitcher_hand"], how="left")
    return base


BAT_COMPONENTS = ["m", "r", "mr", "ob", "oz"]


def count_bucket(frame: pd.DataFrame) -> np.ndarray:
    """볼카운트 3군: 투수우세(0) / 중립(1) / 타자우세(2)."""
    b = pd.to_numeric(frame["balls_before"]).to_numpy()
    s = pd.to_numeric(frame["strikes_before"]).to_numpy()
    return np.where(s > b, 0, np.where(b > s, 2, 1))


def make_count_platoon_table(train: pd.DataFrame, K: int = PLATOON_K) -> pd.DataFrame:
    """카운트별 플래툰 (V19 H2). 2단계 차감이 핵심이다.

        split(p, h, count) = EB(투수 x 타자손 x 카운트군) - EB(투수 x 타자손)

    전역 플래툰을 명시적으로 빼야 새 정보만 남는다. V8 에서 성분별 플래툰이
    리그평균만 빼고 실패한 이유(+0.16)가 전역 플래툰과 중복이었기 때문이다.
    같은 형태인데 2단계로 빼니 +8.44 가 됐다.
    """
    d = pd.DataFrame({"p": train["pitcher_id"].to_numpy(),
                      "h": train["batter_hand"].to_numpy(),
                      "c": count_bucket(train),
                      "y": train[TARGET].to_numpy()})
    league = float(d["y"].mean())
    g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = d.groupby(["p", "h", "c"])["y"].agg(["sum", "size"])
    eb2 = (g2["sum"] + K * league) / (g2["size"] + K)
    eb3 = (g3["sum"] + K * league) / (g3["size"] + K)
    out = eb3.rename("eb3").reset_index()
    out = out.merge(eb2.rename("eb2").reset_index(), on=["p", "h"], how="left")
    out["count_platoon_split"] = out["eb3"] - out["eb2"]
    out["count_platoon_rel"] = (g3["size"].reindex(
        pd.MultiIndex.from_arrays([out["p"], out["h"], out["c"]])).to_numpy()
        / (g3["size"].reindex(
            pd.MultiIndex.from_arrays([out["p"], out["h"], out["c"]])).to_numpy() + K))
    out = out.rename(columns={"p": "pitcher_id", "h": "batter_hand", "c": "count_bucket"})
    out.attrs["league_mean"] = league
    return out[["pitcher_id", "batter_hand", "count_bucket",
                "count_platoon_split", "count_platoon_rel"]]


def inning_bucket(frame: pd.DataFrame) -> np.ndarray:
    """이닝 4군: 1-3 / 4-6 / 7-9 / 연장."""
    return np.digitize(pd.to_numeric(frame["inning"]).to_numpy(), [4, 7, 10])


def make_inning_platoon_table(train: pd.DataFrame, K: int = PLATOON_K) -> pd.DataFrame:
    """이닝별 플래툰 (V22 J_P_hi). 카운트별과 같은 2단계 차감.

        split(p, h, inning) = EB(투수 x 타자손 x 이닝군) - EB(투수 x 타자손)
    """
    d = pd.DataFrame({"p": train["pitcher_id"].to_numpy(),
                      "h": train["batter_hand"].to_numpy(),
                      "i": inning_bucket(train),
                      "y": train[TARGET].to_numpy()})
    league = float(d["y"].mean())
    g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = d.groupby(["p", "h", "i"])["y"].agg(["sum", "size"])
    eb2 = (g2["sum"] + K * league) / (g2["size"] + K)
    eb3 = (g3["sum"] + K * league) / (g3["size"] + K)
    out = eb3.rename("eb3").reset_index().merge(
        eb2.rename("eb2").reset_index(), on=["p", "h"], how="left")
    idx = pd.MultiIndex.from_arrays([out["p"], out["h"], out["i"]])
    sz = g3["size"].reindex(idx).to_numpy()
    out["inning_platoon_split"] = out["eb3"] - out["eb2"]
    out["inning_platoon_rel"] = sz / (sz + K)
    out = out.rename(columns={"p": "pitcher_id", "h": "batter_hand",
                              "i": "inning_bucket"})
    return out[["pitcher_id", "batter_hand", "inning_bucket",
                "inning_platoon_split", "inning_platoon_rel"]]


def build(frame: pd.DataFrame, spec: dict, platoon: pd.DataFrame,
          bat_platoon: pd.DataFrame | None = None,
          count_platoon: pd.DataFrame | None = None,
          inning_platoon: pd.DataFrame | None = None) -> pd.DataFrame:
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

    if bat_platoon is not None:
        bkey = pd.MultiIndex.from_arrays(
            [pd.to_numeric(x["batter_id"]), pd.to_numeric(x["pitcher_hand"])])
        bt = bat_platoon.set_index(["batter_id", "pitcher_hand"]).reindex(bkey)
        bsp = bt["bat_platoon_split"].fillna(0.0).to_numpy(np.float64)
        brel = bt["bat_platoon_rel"].fillna(0.0).to_numpy(np.float64)
        out["bat_platoon_split"] = bsp
        out["bat_platoon_rel"] = brel
        out["bat_platoon_split_w"] = bsp * brel
        for tag in BAT_COMPONENTS:
            out[f"bat_pl_{tag}"] = bt[f"bat_pl_{tag}"].fillna(0.0).to_numpy(np.float64)

    if count_platoon is not None:
        ckey = pd.MultiIndex.from_arrays(
            [pd.to_numeric(x["pitcher_id"]), pd.to_numeric(x["batter_hand"]),
             count_bucket(x)])
        ct = count_platoon.set_index(
            ["pitcher_id", "batter_hand", "count_bucket"]).reindex(ckey)
        csp = ct["count_platoon_split"].fillna(0.0).to_numpy(np.float64)
        crel = ct["count_platoon_rel"].fillna(0.0).to_numpy(np.float64)
        out["count_platoon_split"] = csp
        out["count_platoon_rel"] = crel
        out["count_platoon_w"] = csp * crel

    if inning_platoon is not None:
        ikey = pd.MultiIndex.from_arrays(
            [pd.to_numeric(x["pitcher_id"]), pd.to_numeric(x["batter_hand"]),
             inning_bucket(x)])
        it = inning_platoon.set_index(
            ["pitcher_id", "batter_hand", "inning_bucket"]).reindex(ikey)
        isp = it["inning_platoon_split"].fillna(0.0).to_numpy(np.float64)
        irel = it["inning_platoon_rel"].fillna(0.0).to_numpy(np.float64)
        out["inning_platoon_split"] = isp
        out["inning_platoon_rel"] = irel
        out["inning_platoon_w"] = isp * irel

    return pd.DataFrame(out)


def matrix(frame: pd.DataFrame, spec: dict, platoon: pd.DataFrame) -> np.ndarray:
    """spec['columns'] 순서로 float32 행렬을 만든다."""
    d = build(frame, spec, platoon)
    return d[spec["columns"]].to_numpy(np.float32)
