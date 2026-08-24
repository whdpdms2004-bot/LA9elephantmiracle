"""Pitch Predict(Josh Mancuso) 아이디어 차용 피처 — 전부 투구 전 시점 안전(shift(1) expanding).

추가 그룹
  1) batter_scout : 타자 스카우팅 리포트 (구종 카테고리별 chase/whiff/taken-strike/상대빈도)
  2) gameflow     : 직전 3구 상태(존/스윙/체이스) + 최근 5·15구 카테고리%·스트라이크%
  3) prior_ab     : 직전 타석 결과 플래그(볼넷/삼진/안타/홈런/실점 직후)
  4) lineup       : 타순 슬롯 근사

구종 카테고리: fb(FF,SI,FC) / br(SL,CU,KC,ST,SC) / off(CH,FS,KN,EP)
"""
from __future__ import annotations
import numpy as np, pandas as pd

FB = {"FF", "SI", "FC", "FT"}
BR = {"SL", "CU", "KC", "ST", "SC", "SV", "CS"}
OFF = {"CH", "FS", "KN", "EP", "FO"}
CATS = ["fb", "br", "off"]
SEQ = ["game_date", "game_pk", "at_bat_number", "pitch_number"]


def pitch_category(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    return pd.Series(np.select([s.isin(FB), s.isin(BR), s.isin(OFF)], ["fb", "br", "off"],
                               default="other"), index=s.index, dtype="object")


def _expand_rate(df, key, num, den, name, lam, mu):
    """key별 expanding(shift1) 비율 + Beta-Binomial 수축. 분모 컬럼도 반환."""
    g = df.groupby(key, sort=False)
    n = g[num].cumsum().groupby(df[key]).shift(1)
    d = g[den].cumsum().groupby(df[key]).shift(1)
    df[name] = ((n.fillna(0) + lam * mu) / (d.fillna(0) + lam)).astype("float32")
    df[f"{name}_n"] = d.fillna(0).astype("float32")
    return [name, f"{name}_n"]


def add_batter_scout(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """타자의 구종 카테고리별 성향 (career expanding, shift(1))."""
    df = df.copy()
    cat = pitch_category(df["pitch_type"])
    made = []
    for c in CATS:
        m = (cat == c).astype("float32")
        df[f"_c_{c}"] = m
        df[f"_c_{c}_sw"] = (m * df["is_swing"]).astype("float32")
        df[f"_c_{c}_wh"] = (m * df["is_whiff"]).astype("float32")
        df[f"_c_{c}_tk"] = (m * df["is_take"]).astype("float32")
        df[f"_c_{c}_cs"] = (m * df["is_called"]).astype("float32")
        df[f"_c_{c}_oz"] = (m * df["out_zone"]).astype("float32")
        df[f"_c_{c}_ch"] = (m * df["out_zone"] * df["is_swing"]).astype("float32")
    df["_one"] = 1.0
    mu_sw = float(df["is_swing"].mean()); mu_wh = float(df["is_whiff"].mean())
    mu_cs = float(df["is_called"].mean()); mu_ch = 0.28
    for c in CATS:
        made += _expand_rate(df, "batter", f"_c_{c}", "_one", f"b_faced_{c}", 200, 1/3)
        made += _expand_rate(df, "batter", f"_c_{c}_wh", f"_c_{c}_sw", f"b_whiff_{c}", 100, mu_wh)
        made += _expand_rate(df, "batter", f"_c_{c}_cs", f"_c_{c}_tk", f"b_cstrike_{c}", 100, mu_cs)
        made += _expand_rate(df, "batter", f"_c_{c}_ch", f"_c_{c}_oz", f"b_chase_{c}", 80, mu_ch)
        made += _expand_rate(df, "batter", f"_c_{c}_sw", f"_c_{c}", f"b_swing_{c}", 100, mu_sw)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_c_")] + ["_one"], errors="ignore")
    return df, made


def add_gameflow(df: pd.DataFrame, windows=(5, 15)) -> tuple[pd.DataFrame, list[str]]:
    """직전 3구 상태 + 최근 N구 카테고리%/스트라이크% (투수 기준, 경기 내)."""
    df = df.copy()
    cat = pitch_category(df["pitch_type"])
    g = df.groupby(["game_pk", "pitcher"], sort=False)
    made = []
    # 직전 3구: 존 여부 / 스윙 / 체이스 / CSW
    for lag in (1, 2, 3):
        for src, nm in [("in_zone", "inzone"), ("is_swing", "swung"), ("is_csw", "csw")]:
            col = f"prev_{nm}_{lag}"
            df[col] = g[src].shift(lag).astype("float32"); made.append(col)
    # 최근 N구 카테고리 비율 + 스트라이크율 (투수 전체 시퀀스 기준, cumsum 차분)
    pit = df["pitcher"]
    tmp = {f"_cat_{c}": (cat == c).astype("float32") for c in CATS}
    tmp["_stk"] = df["is_strike"].astype("float32")
    tmp["_one"] = 1.0
    for k, v in tmp.items(): df[k] = v
    for N in windows:
        den = None
        for k in ["_one"] + [f"_cat_{c}" for c in CATS] + ["_stk"]:
            cs = df.groupby(pit, sort=False)[k].cumsum()
            last = cs.groupby(pit).shift(1) - cs.groupby(pit).shift(N + 1).fillna(0.0)
            if k == "_one": den = last.replace(0, np.nan); continue
            nm = f"p_l{N}_{k.replace('_cat_','cat_').strip('_')}"
            df[nm] = (last / den).astype("float32"); made.append(nm)
    df = df.drop(columns=list(tmp), errors="ignore")
    return df, made


def add_prior_ab(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """직전 타석 결과 플래그 (같은 경기·투수 기준)."""
    df = df.copy()
    ev = df["events"].astype("string") if "events" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    ab = df.groupby(["game_pk", "pitcher", "at_bat_number"], sort=False)
    last_ev = ev.groupby([df["game_pk"], df["pitcher"], df["at_bat_number"]]).transform("last")
    pa = df[["game_pk", "pitcher", "at_bat_number"]].copy(); pa["ev"] = last_ev
    pa = pa.drop_duplicates(["game_pk", "pitcher", "at_bat_number"]).sort_values(["game_pk", "pitcher", "at_bat_number"])
    pa["prev_ev"] = pa.groupby(["game_pk", "pitcher"])["ev"].shift(1)
    K = {"strikeout", "strikeout_double_play"}
    BB = {"walk", "intent_walk", "hit_by_pitch"}
    HIT = {"single", "double", "triple", "home_run"}
    pe = pa["prev_ev"]
    pa["after_k"] = pe.isin(K).fillna(False).astype("int8")
    pa["after_bb"] = pe.isin(BB).fillna(False).astype("int8")
    pa["after_hit"] = pe.isin(HIT).fillna(False).astype("int8")
    pa["after_hr"] = pe.eq("home_run").fillna(False).astype("int8")
    made = ["after_k", "after_bb", "after_hit", "after_hr"]
    df = df.merge(pa[["game_pk", "pitcher", "at_bat_number"] + made],
                  on=["game_pk", "pitcher", "at_bat_number"], how="left")
    for c in made: df[c] = df[c].fillna(0).astype("int8")
    return df, made


def add_lineup(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """타순 슬롯 근사: 팀별 타석 순서를 9로 나눈 나머지."""
    df = df.copy()
    side = np.where(df["inning_topbot"].eq("Top"), "away", "home")
    key = df["game_pk"].astype(str) + "_" + side
    order = df.groupby([key, "at_bat_number"], sort=False).ngroup()
    rank = pd.Series(order, index=df.index).groupby(key).rank(method="dense")
    df["lineup_slot"] = (((rank - 1) % 9) + 1).astype("float32")
    return df, ["lineup_slot"]


def add_batter_history(df: pd.DataFrame, windows=(14,), lastn=()) -> tuple[pd.DataFrame, list[str]]:
    """타자 측 6지표를 창별로 (2주 / 시즌 / 지난시즌 / 전기간 / 최근 N구) + 투수 좌우 스플릿.
    전부 shift(1) 시간안전. 지표별 분모 분리 + 표본수 포함."""
    df = df.copy()
    one = np.float32(1.0)
    num_den = {
        "cswA": (df["is_csw"].astype("float32"), np.full(len(df), one)),          # 허용 CSW율
        "whiffB": (df["is_whiff"].astype("float32"), df["is_swing"].astype("float32")),
        "calledB": (df["is_called"].astype("float32"), df["is_take"].astype("float32")),
        "chaseB": ((df["is_swing"] * df["out_zone"]).astype("float32"), df["out_zone"].astype("float32")),
        "swingB": (df["is_swing"].astype("float32"), np.full(len(df), one)),
    }   # 메모리 제약(3GB)으로 4지표만
    cols = []
    for k, (n, d) in num_den.items():
        df[f"_bn_{k}"] = n; df[f"_bd_{k}"] = d; cols += [f"_bn_{k}", f"_bd_{k}"]
    mu = {k: float(np.asarray(n).sum() / max(np.asarray(d).sum(), 1)) for k, (n, d) in num_den.items()}
    lam = {"cswA": 300, "whiffB": 200, "calledB": 200, "swingB": 300, "chaseB": 150}
    made = []
    import gc

    # --- 전부 벡터화(cumsum): 시즌누적 / 전기간 / 최근 N구 (메모리 절약) ---
    bt = df["batter"]; byr = [df["batter"], df["game_year"]]
    for c in cols:
        cs_y = df.groupby(byr, sort=False)[c].cumsum()
        df[f"{c}__bszn"] = cs_y.groupby(byr).shift(1).astype("float32")
        cs = df.groupby(bt, sort=False)[c].cumsum()
        p1 = cs.groupby(bt).shift(1)
        df[f"{c}__bcar"] = p1.astype("float32")
        for N in (200,):
            df[f"{c}__bl{N}"] = (p1 - cs.groupby(bt).shift(N + 1).fillna(0.0)).astype("float32")
        del cs_y, cs, p1
    gc.collect()
    # --- 지난 시즌 전체 ---
    ps = df.groupby(["batter", "game_year"])[cols].sum().reset_index()
    ps["game_year"] = ps["game_year"] + 1
    ps = ps.rename(columns={c: f"{c}__bpszn" for c in cols})
    df = df.merge(ps, on=["batter", "game_year"], how="left")
    del ps; gc.collect()
    # --- 비율화 ---
    for w in ["bszn", "bcar", "bpszn", "bl200"]:
        for k in num_den:
            n, d_ = df.get(f"_bn_{k}__{w}"), df.get(f"_bd_{k}__{w}")
            if n is None or d_ is None: continue
            f = f"b_{k}_{w}"
            df[f] = ((n.fillna(0) + lam[k] * mu[k]) / (d_.fillna(0) + lam[k])).astype("float32")
            df[f"{f}_n"] = d_.fillna(0).astype("float32")
            made += [f, f"{f}_n"]
    # --- 투수 좌우 스플릿 (career expanding) ---
    for hand in ["R", "L"]:
        m = df["p_throws"].eq(hand)
        df["_h_n"] = (df["is_csw"] * m).astype("float32"); df["_h_d"] = m.astype("float32")
        g = df.groupby("batter", sort=False)
        n = g["_h_n"].cumsum().groupby(df["batter"]).shift(1)
        d_ = g["_h_d"].cumsum().groupby(df["batter"]).shift(1)
        f = f"b_csw_vs_{hand}HP"
        df[f] = ((n.fillna(0) + 200 * mu["cswA"]) / (d_.fillna(0) + 200)).astype("float32")
        made.append(f)
    drop = [c for c in df.columns if c.startswith("_bn_") or c.startswith("_bd_") or c in ("_h_n", "_h_d")]
    df = df.drop(columns=drop, errors="ignore")
    gc.collect()
    return df, made


def add_workload(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """피로/워크로드: 오늘 몇 개 던졌나 · 평소 한 경기 몇 개 · 평소 대비 지금 몇 %.

    누수 주의: 현재 경기의 '총' 투구수는 미래 정보 → 평균은 **과거 등판만**으로 계산(shift).
    오늘 현재까지 투구수(pitcher_pitch_count_before)는 투구 전 확정이라 사용 가능.
    """
    df = df.copy()
    today = pd.to_numeric(df.get("pitcher_pitch_count_before"), errors="coerce").astype("float32")
    if today.isna().all():
        g = df.groupby(["game_pk", "pitcher"], sort=False)
        today = g.cumcount().astype("float32")
    df["p_pitches_today"] = today

    # 등판(경기)별 총 투구수 → 투수별 과거 평균/직전값 (전부 shift)
    games = (df.groupby(["pitcher", "game_pk", "game_date"], sort=False).size()
               .reset_index(name="_n").sort_values(["pitcher", "game_date", "game_pk"]))
    gg = games.groupby("pitcher", sort=False)["_n"]
    games["p_game_pitch_avg_car"] = gg.transform(lambda s: s.shift(1).expanding().mean()).astype("float32")
    games["p_game_pitch_avg_l5"] = gg.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).astype("float32")
    games["p_prev_game_pitches"] = gg.shift(1).astype("float32")
    games["p_game_pitch_sd_car"] = gg.transform(lambda s: s.shift(1).expanding().std()).astype("float32")
    games["p_outing_index"] = gg.cumcount().astype("float32")          # 몇 번째 등판인가
    # 시즌 누적 투구수(직전 등판까지)
    yr = games["game_date"].dt.year if np.issubdtype(games["game_date"].dtype, np.datetime64) else None
    if yr is not None:
        games["_yr"] = yr
        games["p_season_pitch_cum"] = (games.groupby(["pitcher", "_yr"], sort=False)["_n"]
                                       .transform(lambda s: s.shift(1).cumsum()).astype("float32"))
        games = games.drop(columns=["_yr"])
    cols = ["p_game_pitch_avg_car", "p_game_pitch_avg_l5", "p_prev_game_pitches",
            "p_game_pitch_sd_car", "p_outing_index"] + (["p_season_pitch_cum"] if yr is not None else [])
    df = df.merge(games[["pitcher", "game_pk"] + cols], on=["pitcher", "game_pk"], how="left")

    # 평소 대비 현재 (핵심 신호)
    avg = df["p_game_pitch_avg_car"].replace(0, np.nan)
    df["p_workload_ratio"] = (df["p_pitches_today"] / avg).astype("float32")          # 1.0 = 평소만큼
    df["p_workload_excess"] = (df["p_pitches_today"] - avg).astype("float32")         # 초과 개수
    df["p_over_usual"] = (df["p_workload_ratio"] > 1.0).astype("int8")
    sd = df["p_game_pitch_sd_car"].replace(0, np.nan)
    df["p_workload_z"] = ((df["p_pitches_today"] - avg) / sd).astype("float32")       # 표준화 초과
    avg5 = df["p_game_pitch_avg_l5"].replace(0, np.nan)
    df["p_workload_ratio_l5"] = (df["p_pitches_today"] / avg5).astype("float32")
    made = ["p_pitches_today"] + cols + ["p_workload_ratio", "p_workload_excess",
                                         "p_over_usual", "p_workload_z", "p_workload_ratio_l5"]
    return df, made


def add_all(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.sort_values(SEQ, kind="stable").reset_index(drop=True)
    groups = {}
    df, groups["batter_scout"] = add_batter_scout(df)
    df, groups["gameflow"] = add_gameflow(df)
    df, groups["prior_ab"] = add_prior_ab(df)
    df, groups["lineup"] = add_lineup(df)
    return df, groups   # batter_hist는 메모리 제약상 2단계(prep_g.py)에서 별도 추가
