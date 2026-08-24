"""
csw_pipeline.py — CSW(콜드스트라이크+헛스윙) 투구 전 예측 공용 파이프라인.

리뷰 반영(필수 4 + α):
  · 모든 이력/인코딩 = 전체기간 시간정렬 후 shift(1) expanding/rolling
    → train 내부 미래누수 없음 + 2019는 online/prequential(모델 파라미터는 2017–18 고정).
  · 지표별 분자/분모 분리 + 지표별 Beta-Binomial 수축(λ) + 표본수·결측 지시자.
  · Kirby → release-angle repeatability 로 재명명(command 아님).
  · 구종별 아스널(velo/usage) 분리. 물리 n = 해당 측정값 non-null 수.
  · Basic / Basic-history / Historical 분리. is_starter 안전정의. 분산 clamp, ddof=0.
  · 주심(umpire) 미사용.

한 파일로 유지(노트북은 호출+시각화만). 권장 src/ 분할은 보고서 계획 참고.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

RAW = Path(__file__).resolve().parents[2] / "data" / "statcast_2017_2019_raw_csw.parquet"
SEQ = ["game_date", "game_pk", "at_bat_number", "pitch_number"]
TRAIN_YEARS, TEST_YEARS = (2017, 2018), (2019,)

# ── 라벨 매핑 (16개 description 전부 분류) ────────────────────────────────
WHIFF  = {"swinging_strike", "swinging_strike_blocked"}
CALLED = {"called_strike"}
SWING  = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
          "hit_into_play", "foul_bunt", "missed_bunt", "bunt_foul_tip",
          "swinging_pitchout", "foul_pitchout"}
FASTBALL = {"FF", "SI", "FC"}
BREAKING = {"SL", "CU", "KC", "ST"}

# ── 누수 열: 현재 투구 물리/위치/결과 + 미래 정보 ────────────────────────
CURRENT_PITCH = {
    "type","description","des","events","zone","plate_x","plate_z","sz_top","sz_bot",
    "release_speed","effective_speed","release_spin_rate","spin_axis","spin_dir","pfx_x","pfx_z",
    "vx0","vy0","vz0","ax","ay","az","release_pos_x","release_pos_y","release_pos_z","release_extension",
    "hit_location","bb_type","launch_speed","launch_angle","hit_distance_sc","launch_speed_angle",
    "estimated_ba_using_speedangle","estimated_woba_using_speedangle","woba_value","woba_denom",
    "babip_value","iso_value","api_break_z_with_gravity","api_break_x_arm","api_break_x_batter_in",
    "delta_home_win_exp","delta_run_exp","delta_pitcher_run_exp","home_win_exp","bat_win_exp",
    "post_home_score","post_away_score","post_bat_score","post_fld_score",
    "pitch_type","pitch_name",  # 현재 구종(구종 결정 전 예측)
}
FUTURE = {"pitcher_days_until_next_game", "batter_days_until_next_game"}
LABELS = {"is_csw","is_swing","is_whiff","is_called","is_take","in_zone","out_zone","is_strike"}
BANNED = CURRENT_PITCH | FUTURE | LABELS

LOAD_COLS = SEQ + [
    "game_year","game_type","home_team","away_team","pitcher","batter","fielder_2","player_name",
    "stand","p_throws","balls","strikes","outs_when_up","inning","inning_topbot",
    "on_1b","on_2b","on_3b","home_score","away_score","bat_score","fld_score",
    "if_fielding_alignment","of_fielding_alignment",
    "age_pit","age_bat","n_thruorder_pitcher","n_priorpa_thisgame_player_at_bat","pitcher_days_since_prev_game",
    # 아래는 라벨/이력 계산용(현재행 피처로는 금지, 과거집계·라벨에만)
    "pitch_type","description","type","events","zone","release_speed",
    "vx0","vy0","vz0","ax","ay","az","release_pos_y","release_extension",
    "is_csw",
]

RATE_METRICS = ["csw", "whiff", "called", "zone", "chase", "fps"]  # 지표별 분모 상이
DEFAULT_LAMBDA = {"csw":200, "whiff":150, "called":150, "zone":200, "chase":120, "fps":100}
LASTN_WINDOWS = [100, 500]                       # 최근 N'구' rolling
WINDOWS = ["day", "2w", "szn", "pszn", "car"] + [f"l{n}" for n in LASTN_WINDOWS]


# ══════════════════════════════════════════════════════════════════════════
# 1. 로드 + 라벨
# ══════════════════════════════════════════════════════════════════════════
def load_subset(top_pitchers: int = 60, pitcher_ids=None) -> pd.DataFrame:
    """pushdown 필터로 필요한 투수 행만 로드(메모리 안전). top_pitchers=0/None → 전체.
    pitcher_ids 지정 시 해당 투수만(배치 스트리밍용)."""
    import pyarrow.parquet as pq
    avail = set(pq.ParquetFile(RAW).schema_arrow.names)
    use = [c for c in LOAD_COLS if c in avail]
    if pitcher_ids is None and top_pitchers:
        light = pd.read_parquet(RAW, columns=["pitcher", "game_year", "game_type"])
        tr = light[light["game_type"].eq("R") & light["game_year"].isin(TRAIN_YEARS)]
        pitcher_ids = tr.groupby("pitcher").size().sort_values(ascending=False).head(top_pitchers).index.tolist()
        del light
    # 스트리밍 배치 읽기 (parquet는 날짜순 정렬이라 pushdown이 안 먹힘 → 배치별 필터)
    pf = pq.ParquetFile(RAW)
    keep = set(int(x) for x in pitcher_ids) if pitcher_ids is not None else None
    parts = []
    for batch in pf.iter_batches(batch_size=250_000, columns=use):
        b = batch.to_pandas()
        b = b[b["game_type"].eq("R")]
        if keep is not None:
            b = b[b["pitcher"].isin(keep)]
        if len(b):
            parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for c in ["balls","strikes","outs_when_up","inning","home_score","away_score","bat_score",
              "fld_score","age_pit","age_bat","n_thruorder_pitcher",
              "n_priorpa_thisgame_player_at_bat","pitcher_days_since_prev_game","zone","release_speed",
              "vx0","vy0","vz0","ax","ay","az","release_pos_y","release_extension"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in ["game_year","pitcher","batter","game_pk"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
    if top_pitchers:
        tr = df[df["game_year"].isin(TRAIN_YEARS)]
        keep = tr.groupby("pitcher").size().sort_values(ascending=False).head(top_pitchers).index
        df = df[df["pitcher"].isin(keep)].copy()
    df = df.sort_values(SEQ, kind="stable", na_position="last").reset_index(drop=True)
    return add_labels(df)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    d = df["description"].astype("string")
    df["is_swing"]  = d.isin(SWING).astype("int8")
    df["is_whiff"]  = d.isin(WHIFF).astype("int8")
    df["is_called"] = d.isin(CALLED).astype("int8")
    df["is_take"]   = (1 - df["is_swing"]).astype("int8")
    if "is_csw" not in df:
        df["is_csw"] = d.isin(CALLED | WHIFF).astype("int8")
    df["is_csw"] = pd.to_numeric(df["is_csw"], errors="coerce").fillna(0).astype("int8")
    z = pd.to_numeric(df["zone"], errors="coerce").astype("float64")
    df["in_zone"]  = z.between(1, 9).fillna(False).astype("int8")
    df["out_zone"] = z.isin([11, 12, 13, 14]).astype("int8")
    df["is_strike"] = df["type"].eq("S").astype("int8")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2. Base features (현재 상황 + 단순 식별자 + Basic-history: 경기 내/직전)
# ══════════════════════════════════════════════════════════════════════════
def add_base_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    i1, i2, i3 = (df[c].notna().astype("int8") for c in ["on_1b","on_2b","on_3b"])
    df["score_diff_bat"] = (df["bat_score"] - df["fld_score"]).astype("float32")
    df["base_state"] = (i1 + 2*i2 + 4*i3).astype("int8")
    df["runner_count"] = (i1 + i2 + i3).astype("int8")
    df["risp"] = ((i2 == 1) | (i3 == 1)).astype("int8")
    df["bases_loaded"] = ((i1==1)&(i2==1)&(i3==1)).astype("int8")
    df["two_strike"] = df["strikes"].eq(2).astype("int8")
    df["full_count"] = (df["balls"].eq(3) & df["strikes"].eq(2)).astype("int8")
    df["pitcher_ahead"] = ((df["strikes"] > df["balls"]) & ~df["full_count"].astype(bool)).astype("int8")
    df["batter_ahead"] = (df["balls"] > df["strikes"]).astype("int8")
    df["same_handed_matchup"] = df["p_throws"].eq(df["stand"]).astype("int8")
    df["matchup"] = (df["p_throws"].astype("string") + "_" + df["stand"].astype("string"))
    df["late_inning"] = df["inning"].ge(7).astype("int8")
    df["extra_inning"] = df["inning"].ge(10).astype("int8")
    df["close_game"] = df["score_diff_bat"].abs().le(2).astype("int8")
    df["rest_days"] = df["pitcher_days_since_prev_game"].astype("float32")

    # Basic-history: 경기 내 누적/lag (별도 장기테이블 불필요)
    g = df.groupby(["game_pk","pitcher"], sort=False, group_keys=False)
    df["pitcher_pitch_count_before"] = g.cumcount().astype("int32")
    df["prev_pitch_type_1"] = g["pitch_type"].shift(1)
    df["prev_pitch_type_2"] = g["pitch_type"].shift(2)
    df["prev_description"] = g["description"].shift(1)

    # is_starter 안전정의: 그 경기에서 해당 수비팀이 처음 쓴 투수인가(첫 투구 시점 확정)
    df["fld_team"] = np.where(df["inning_topbot"].eq("Top"), df["home_team"], df["away_team"])
    order = df.groupby(["game_pk","fld_team","pitcher"])["inning"].transform("min")  # placeholder
    first_pitcher = (df.sort_values(SEQ).groupby(["game_pk","fld_team"])["pitcher"].transform("first"))
    df["is_starter"] = df["pitcher"].eq(first_pitcher).astype("int8")

    base_num = ["balls","strikes","outs_when_up","inning","score_diff_bat","base_state","runner_count",
                "risp","bases_loaded","two_strike","full_count","pitcher_ahead","batter_ahead",
                "same_handed_matchup","late_inning","extra_inning","close_game",
                "age_pit","age_bat","n_thruorder_pitcher","n_priorpa_thisgame_player_at_bat",
                "rest_days","pitcher_pitch_count_before","is_starter"]
    base_cat = ["stand","p_throws","matchup","inning_topbot","if_fielding_alignment",
                "of_fielding_alignment","home_team","prev_pitch_type_1","prev_pitch_type_2","prev_description"]
    groups = {
        "situation": [c for c in base_num if c not in
                      ("age_pit","age_bat","n_thruorder_pitcher","n_priorpa_thisgame_player_at_bat",
                       "rest_days","pitcher_pitch_count_before","is_starter")]
                     + ["stand","p_throws","matchup","inning_topbot","if_fielding_alignment","of_fielding_alignment","home_team"],
        "count_only": ["balls","strikes","stand","p_throws"],
        "basic_history": ["age_pit","age_bat","n_thruorder_pitcher","n_priorpa_thisgame_player_at_bat",
                          "rest_days","pitcher_pitch_count_before","is_starter",
                          "prev_pitch_type_1","prev_pitch_type_2","prev_description"],
    }
    return df, {"base_num": base_num, "base_cat": base_cat, "groups": groups}


# ══════════════════════════════════════════════════════════════════════════
# 3. History features (창별 지표: 분자/분모 분리 + BB 수축 + 표본·결측)
# ══════════════════════════════════════════════════════════════════════════
def _num_den(df: pd.DataFrame) -> pd.DataFrame:
    one = np.int32(1)
    out = pd.DataFrame(index=df.index)
    out["csw_n"], out["csw_d"] = df["is_csw"], 1
    out["whiff_n"], out["whiff_d"] = df["is_whiff"], df["is_swing"]
    out["called_n"], out["called_d"] = df["is_called"], df["is_take"]
    out["zone_n"], out["zone_d"] = df["in_zone"], 1
    out["chase_n"] = ((df["is_swing"] == 1) & (df["out_zone"] == 1)).astype("int32")
    out["chase_d"] = df["out_zone"]
    fp = (df["balls"].eq(0) & df["strikes"].eq(0)).astype("int32")
    out["fps_n"], out["fps_d"] = (fp & df["is_strike"]).astype("int32"), fp
    return out.astype("float32")   # 메모리 절약(3GB 환경)


def _bb_shrink(num, den, mu, lam):
    return (num + lam * mu) / (den + lam)


def add_history_features(df: pd.DataFrame, lambdas: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """투수 결과성향 6지표 × 4창(day/2w/szn/pszn). 모두 shift(1) 시간안전."""
    lambdas = lambdas or DEFAULT_LAMBDA
    df = df.sort_values(SEQ, kind="stable").reset_index(drop=True)
    nd = _num_den(df)
    df = pd.concat([df, nd], axis=1)
    ncols = [f"{m}_n" for m in RATE_METRICS] + [f"{m}_d" for m in RATE_METRICS]
    league_mu = {m: float(df.loc[df["game_year"].isin(TRAIN_YEARS), f"{m}_n"].sum() /
                            max(df.loc[df["game_year"].isin(TRAIN_YEARS), f"{m}_d"].sum(), 1)) for m in RATE_METRICS}

    feats = {}
    # --- day (경기 내 누적, shift) ---
    g = df.groupby(["game_pk","pitcher"], sort=False, group_keys=False)
    for c in ncols:
        df[f"{c}__day"] = g[c].transform(lambda s: s.shift(1).cumsum())
    # --- 2w / szn (일자 집계 후 시간 롤링) ---
    daily = df.groupby(["pitcher","game_year","game_date"])[ncols].sum().reset_index()
    daily = daily.sort_values(["pitcher","game_date"])
    def _pp(gp):
        gp = gp.set_index("game_date").sort_index()
        w2 = gp[ncols].rolling("14D", closed="left").sum()
        sz = gp.groupby(gp["game_year"])[ncols].cumsum().groupby(gp["game_year"]).shift(1)
        car = gp[ncols].cumsum().shift(1)          # 전 기간(career) 누적, 당일 제외
        out = gp[["game_year"]].copy()
        for c in ncols:
            out[f"{c}__2w"], out[f"{c}__szn"], out[f"{c}__car"] = w2[c], sz[c], car[c]
        return out.reset_index()
    dwin = daily.groupby("pitcher", group_keys=True).apply(_pp)
    dwin = dwin[[c for c in dwin.columns if not c.startswith("level_")]].reset_index()
    dwin = dwin[[c for c in dwin.columns if c in (["pitcher","game_date","game_year"] +
                 [f"{c}__2w" for c in ncols] + [f"{c}__szn" for c in ncols] +
                 [f"{c}__car" for c in ncols])]]
    df = df.merge(dwin, on=["pitcher","game_date","game_year"], how="left")

    # --- 최근 N'구' rolling (시간이 아닌 투구 수 기준; cumsum 차분으로 벡터화) ---
    import gc
    pit = df["pitcher"]
    for c in ncols:
        cs = df.groupby(pit, sort=False)[c].cumsum()
        prev1 = cs.groupby(pit).shift(1)
        for N in LASTN_WINDOWS:
            df[f"{c}__l{N}"] = (prev1 - cs.groupby(pit).shift(N + 1).fillna(0.0)).astype("float32")
        del cs, prev1
    gc.collect()
    # --- pszn (지난 시즌 전체) ---
    ps = df.groupby(["pitcher","game_year"])[ncols].sum().reset_index()
    ps["game_year"] = ps["game_year"] + 1
    ps = ps.rename(columns={c: f"{c}__pszn" for c in ncols})
    df = df.merge(ps, on=["pitcher","game_year"], how="left")

    # --- 지표별 rate + BB 수축 + 표본수 + 결측 ---
    made = []
    for win in WINDOWS:
        for m in RATE_METRICS:
            num, den = df[f"{m}_n__{win}"], df[f"{m}_d__{win}"]
            rate = f"p_{m}_rate_{win}"; df[rate] = _bb_shrink(num.fillna(0), den.fillna(0), league_mu[m], lambdas[m]).astype("float32")
            dcol = f"p_{m}_den_{win}"; df[dcol] = den.fillna(0).astype("float32")
            miss = f"p_{m}_miss_{win}"; df[miss] = den.isna().astype("int8") if win in ("2w","szn","pszn") else (den.fillna(0) == 0).astype("int8")
            made += [rate, dcol, miss]
    feats["pitcher_hist"] = made

    df = df.drop(columns=[c for c in df.columns if any(c.startswith(p) and ("__" in c) for p in [f"{m}_n" for m in RATE_METRICS]+[f"{m}_d" for m in RATE_METRICS])])
    df = df.drop(columns=[f"{m}_{s}" for m in RATE_METRICS for s in ("n","d")], errors="ignore")
    gc.collect()

    # --- 아스널(구종별 velo/usage) + release repeatability : szn/pszn ---
    df, arse = _add_arsenal(df)
    df, rep = _add_release_repeatability(df)
    feats["arsenal"], feats["release_rep"] = arse, rep
    return df, feats


def _daily_roll(df, value_cols):
    """(pitcher,date) 일자합 → 2w(14D rolling)/szn(cumsum shift)/car(전기간)/pszn(전시즌합)."""
    daily = df.groupby(["pitcher","game_year","game_date"])[value_cols].sum().reset_index().sort_values(["pitcher","game_date"])
    def _pp(gp):
        gp = gp.set_index("game_date").sort_index()
        w2 = gp[value_cols].rolling("14D", closed="left").sum()
        sz = gp.groupby(gp["game_year"])[value_cols].cumsum().groupby(gp["game_year"]).shift(1)
        car = gp[value_cols].cumsum().shift(1)
        out = gp[["game_year"]].copy()
        for c in value_cols:
            out[f"{c}__2w"], out[f"{c}__szn"], out[f"{c}__car"] = w2[c], sz[c], car[c]
        return out.reset_index()
    d = daily.groupby("pitcher", group_keys=True).apply(_pp)
    d = d[[c for c in d.columns if not c.startswith("level_")]].reset_index()
    keep = (["pitcher","game_date","game_year"] + [f"{c}__szn" for c in value_cols]
            + [f"{c}__2w" for c in value_cols] + [f"{c}__car" for c in value_cols])
    d = d[[c for c in d.columns if c in keep]]
    ps = df.groupby(["pitcher","game_year"])[value_cols].sum().reset_index()
    ps["game_year"] = ps["game_year"] + 1
    ps = ps.rename(columns={c: f"{c}__pszn" for c in value_cols})
    return d, ps


def _mask(s):  # nullable boolean → numpy bool
    return s.fillna(False).to_numpy()

def _add_arsenal(df):
    speed = pd.to_numeric(df["release_speed"], errors="coerce")
    sp, spok = speed.to_numpy(), speed.notna().to_numpy()
    pt = df["pitch_type"].astype("string")
    cols = {}
    for T in ["FF","SI","SL","CH","CU","FC"]:
        m = _mask(pt.eq(T))
        cols[f"velo_{T}_sum"] = np.where(m & spok, sp, 0.0)
        cols[f"velo_{T}_cnt"] = (m & spok).astype("float64")
    fb, br, pc = _mask(pt.isin(FASTBALL)), _mask(pt.isin(BREAKING)), _mask(pt.notna())
    cols["fb_cnt"] = fb.astype("float64")
    cols["fb_velo_sum"] = np.where(fb & spok, sp, 0.0)
    cols["br_cnt"] = br.astype("float64")
    cols["pitch_cnt"] = pc.astype("float64")
    tmp = pd.DataFrame(cols, index=df.index)
    df = pd.concat([df, tmp], axis=1)
    val = list(cols)
    dsz, dps = _daily_roll(df, val)
    df = df.merge(dsz, on=["pitcher","game_date","game_year"], how="left").merge(dps, on=["pitcher","game_year"], how="left")
    made = []
    for win in ["2w","szn","car","pszn"]:
        for T in ["FF","SI","SL","CH"]:
            f = f"p_{T.lower()}_velo_{win}"; df[f] = (df[f"velo_{T}_sum__{win}"] / df[f"velo_{T}_cnt__{win}"].replace(0, np.nan)).astype("float32"); made.append(f)
        f = f"p_fastball_velo_{win}"; df[f] = (df[f"fb_velo_sum__{win}"] / df[f"fb_cnt__{win}"].replace(0, np.nan)).astype("float32"); made.append(f)
        f = f"p_fastball_usage_{win}"; df[f] = (df[f"fb_cnt__{win}"] / df[f"pitch_cnt__{win}"].replace(0, np.nan)).astype("float32"); made.append(f)
        f = f"p_breaking_usage_{win}"; df[f] = (df[f"br_cnt__{win}"] / df[f"pitch_cnt__{win}"].replace(0, np.nan)).astype("float32"); made.append(f)
        f = f"p_ff_n_{win}"; df[f] = df[f"velo_FF_cnt__{win}"].fillna(0).astype("float32"); made.append(f)
    df["p_fastball_velo_trend"] = (df["p_fastball_velo_szn"] - df["p_fastball_velo_pszn"]).astype("float32"); made.append("p_fastball_velo_trend")
    df = df.drop(columns=[c for c in df.columns if c.endswith("__szn") or c.endswith("__pszn")] + val, errors="ignore")
    return df, made


def _release_angles(df):
    """VRA/HRA(도). vx0..az + release_pos_y/extension로 릴리스 시점 각도 근사."""
    need = ["vx0","vy0","vz0","ax","ay","az"]
    if not all(c in df for c in need): return None, None
    y0, mound = 50.0, 60.5
    yr = pd.to_numeric(df.get("release_pos_y"), errors="coerce")
    if "release_extension" in df: yr = yr.fillna(mound - pd.to_numeric(df["release_extension"], errors="coerce"))
    yr = yr.fillna(54.5)
    vy0, ay = pd.to_numeric(df["vy0"], errors="coerce"), pd.to_numeric(df["ay"], errors="coerce")
    dy = yr - y0
    a, b, c = 0.5*ay, vy0, -dy
    disc = (b*b - 4*a*c).clip(lower=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (-b - np.sqrt(disc)) / (2*a)
        t = t.where(a.abs() > 1e-9, -c/b)
    vx = pd.to_numeric(df["vx0"], errors="coerce") + pd.to_numeric(df["ax"], errors="coerce")*t
    vz = pd.to_numeric(df["vz0"], errors="coerce") + pd.to_numeric(df["az"], errors="coerce")*t
    vyr = vy0 + ay*t
    return np.degrees(np.arctan2(vz, -vyr)), np.degrees(np.arctan2(vx, -vyr))


def _add_release_repeatability(df):
    """FF 릴리스 각도 반복성(과거 SD). 'command' 아님 — 릴리스 산포 지표."""
    vra, hra = _release_angles(df)
    if vra is None: return df, []
    ff = _mask(df["pitch_type"].eq("FF"))
    df["_vra"] = np.where(ff, np.asarray(vra, float), np.nan)
    df["_hra"] = np.where(ff, np.asarray(hra, float), np.nan)
    for c in ["_vra","_hra"]:
        df[f"{c}_sq"] = df[c]**2
        df[f"{c}_cnt"] = df[c].notna().astype("float64")
        df[c] = df[c].fillna(0.0)
    val = ["_vra","_hra","_vra_sq","_hra_sq","_vra_cnt","_hra_cnt"]
    dsz, dps = _daily_roll(df, val)
    df = df.merge(dsz, on=["pitcher","game_date","game_year"], how="left").merge(dps, on=["pitcher","game_year"], how="left")
    made = []
    for win in ["szn","pszn"]:
        for ax in ["vra","hra"]:
            n = df[f"_{ax}_cnt__{win}"]; s = df[f"_{ax}__{win}"]; sq = df[f"_{ax}_sq__{win}"]
            var = np.maximum(sq / n.replace(0, np.nan) - (s / n.replace(0, np.nan))**2, 0.0)  # 음수 clamp, ddof=0
            f = f"p_relangle_{ax}_sd_{win}"; df[f] = np.sqrt(var).astype("float32"); made.append(f)
        disp = np.sqrt(df[f"p_relangle_vra_sd_{win}"]**2 + df[f"p_relangle_hra_sd_{win}"]**2)
        f = f"p_release_repeatability_{win}"; df[f] = (1.0/(1.0+disp)).astype("float32"); made.append(f)
        f = f"p_ff_n_rep_{win}"; df[f] = df[f"_vra_cnt__{win}"].fillna(0).astype("float32"); made.append(f)
        f = f"p_relangle_miss_{win}"; df[f] = (df[f"_vra_cnt__{win}"].fillna(0) < 25).astype("int8"); made.append(f)
    df = df.drop(columns=[c for c in df.columns if c.endswith("__szn") or c.endswith("__pszn")] + val, errors="ignore")
    return df, made


# ══════════════════════════════════════════════════════════════════════════
# 4. Expanding target encoding (시간안전 + prequential) — global 모델 ID용
# ══════════════════════════════════════════════════════════════════════════
def add_target_encoding(df: pd.DataFrame, key: str, lam: float = 200.0, name: str | None = None) -> str:
    df.sort_values(SEQ, kind="stable", inplace=True)
    mu = float(df.loc[df["game_year"].isin(TRAIN_YEARS), "is_csw"].mean())
    g = df.groupby(key, sort=False)
    prior_sum = g["is_csw"].cumsum().shift(1)
    prior_n = g.cumcount()
    col = name or f"{key}_te"
    df[col] = ((prior_sum.fillna(0) + lam*mu) / (prior_n + lam)).astype("float32")
    return col


# ══════════════════════════════════════════════════════════════════════════
# 5. 매트릭스 + 누수 assert
# ══════════════════════════════════════════════════════════════════════════
def build_matrix(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    leaked = BANNED & set(feat_cols)
    assert not leaked, f"누수 열 포함: {sorted(leaked)}"
    import re
    X = df[feat_cols].copy()
    cat = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "string"]
    if cat:
        X = pd.get_dummies(X, columns=cat, dummy_na=True)
    X.columns = [re.sub(r"[\[\]<>,\s]+", "_", str(c)) for c in X.columns]  # XGBoost 안전 이름
    return X.astype("float32")


# ══════════════════════════════════════════════════════════════════════════
# 6. 평가 · 베이스라인
# ══════════════════════════════════════════════════════════════════════════
def ece(y, p, bins=10):
    y = np.asarray(y); p = np.asarray(p); edges = np.linspace(0,1,bins+1)
    idx = np.clip(np.digitize(p, edges)-1, 0, bins-1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean()*abs(y[m].mean()-p[m].mean())
    return float(e)

def metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
    y = np.asarray(y).astype(int); p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    return dict(n=int(len(y)), base=round(float(y.mean()),4),
                roc_auc=round(roc_auc_score(y,p),4) if y.min()!=y.max() else float("nan"),
                pr_auc=round(average_precision_score(y,p),4),
                logloss=round(log_loss(y,p,labels=[0,1]),4),
                brier=round(brier_score_loss(y,p),4), ece=round(ece(y,p),4))

def baselines(df, tr_mask, te_mask):
    y_tr, y_te = df.loc[tr_mask,"is_csw"].to_numpy(), df.loc[te_mask,"is_csw"].to_numpy()
    out = {}
    league = float(y_tr.mean())
    out["league_mean"] = metrics(y_te, np.full(len(y_te), league))
    # count-only: P(CSW|balls,strikes,stand,p_throws) — train 집계 매핑(참조 베이스라인)
    key = ["balls","strikes","stand","p_throws"]
    gm = df[tr_mask].groupby(key)["is_csw"].mean()
    p_te = df.loc[te_mask, key].merge(gm.rename("p"), on=key, how="left")["p"].fillna(league).to_numpy()
    out["count_only"] = metrics(y_te, p_te)
    if "pitcher_te" in df:
        out["pitcher_te"] = metrics(y_te, df.loc[te_mask,"pitcher_te"].fillna(league).to_numpy())
    return out


# ══════════════════════════════════════════════════════════════════════════
# 7. 모델 zoo · Optuna(2017 fit / 2018 val) · SHAP
# ══════════════════════════════════════════════════════════════════════════
def model_zoo(seed=0):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {
        "logreg": Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                            ("clf",LogisticRegression(max_iter=1000, C=1.0))]),
        "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
                    min_samples_leaf=200, l2_regularization=1.0, random_state=seed),
        "lgbm": LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                    min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1),
        "xgb": XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=seed, n_jobs=-1, tree_method="hist", eval_metric="logloss"),
    }

def compare_models(X, y, tr, va, seed=0):
    rows = {}
    for name, mdl in model_zoo(seed).items():
        mdl.fit(X[tr], y[tr]); p = mdl.predict_proba(X[va])[:,1]
        rows[name] = metrics(y[va], p)
    return pd.DataFrame(rows).T

def optuna_tune_lgbm(X, y, tr, va, n_trials=15, seed=0):
    import optuna
    from lightgbm import LGBMClassifier
    from sklearn.metrics import average_precision_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def obj(t):
        params = dict(
            n_estimators=t.suggest_int("n_estimators",150,400,step=50),
            learning_rate=t.suggest_float("learning_rate",0.03,0.12,log=True),
            num_leaves=t.suggest_int("num_leaves",31,127,log=True),
            min_child_samples=t.suggest_int("min_child_samples",100,500,log=True),
            subsample=t.suggest_float("subsample",0.6,1.0),
            colsample_bytree=t.suggest_float("colsample_bytree",0.6,1.0),
            reg_lambda=t.suggest_float("reg_lambda",1e-2,10,log=True),
            reg_alpha=t.suggest_float("reg_alpha",1e-3,5,log=True))
        m = LGBMClassifier(**params, random_state=seed, n_jobs=-1, verbose=-1)
        m.fit(X[tr], y[tr]); p = m.predict_proba(X[va])[:,1]
        return average_precision_score(y[va], p)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return st

def map_columns_to_groups(matrix_cols, feature_group_map: dict) -> dict:
    """get_dummies 후 컬럼(orig_cat_value 포함)을 원 피처 그룹으로 매핑."""
    origs = sorted(feature_group_map, key=len, reverse=True)
    out = {}
    for col in matrix_cols:
        if col in feature_group_map:
            out[col] = feature_group_map[col]; continue
        g = "other"
        for o in origs:
            if col == o or col.startswith(o + "_"):
                g = feature_group_map[o]; break
        out[col] = g
    return out

def shap_by_category(model, X_sample, group_of: dict):
    import shap
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X_sample)
    if isinstance(sv, list): sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3: sv = sv[:, :, -1]
    mean_abs = np.abs(sv).mean(axis=0)
    feat = pd.Series(mean_abs, index=X_sample.columns).sort_values(ascending=False)
    cats = {}
    for f, v in feat.items():
        g = group_of.get(f, "other")
        cats[g] = cats.get(g, 0.0) + float(v)
    catser = pd.Series(cats).sort_values(ascending=False)
    return feat, catser


# ══════════════════════════════════════════════════════════════════════════
# 8. 단계적 ablation
# ══════════════════════════════════════════════════════════════════════════
def staged_ablation(df, stages, tr, te, seed=0):
    from lightgbm import LGBMClassifier
    y = df["is_csw"].to_numpy(); rows = {}; cum = []
    for name, cols in stages:
        cum = sorted(set(cum) | set([c for c in cols if c in df.columns]))
        X = build_matrix(df, cum)
        m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
        m.fit(X.iloc[tr], y[tr]); p = m.predict_proba(X.iloc[te])[:,1]
        rows[name] = {**metrics(y[te], p), "n_feats": X.shape[1]}
    return pd.DataFrame(rows).T


# ══════════════════════════════════════════════════════════════════════════
# 9. 투수별 모델 + 폴백
# ══════════════════════════════════════════════════════════════════════════
def per_pitcher_eval(df, feat_cols, tr, te, min_train=4000, seed=0, global_fit_cap=None):
    from lightgbm import LGBMClassifier
    y = df["is_csw"].to_numpy()
    Xall = build_matrix(df, feat_cols)
    gmodel = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
    tr_pos = np.where(tr)[0]
    if global_fit_cap and len(tr_pos) > global_fit_cap:
        tr_pos = np.random.default_rng(seed).choice(tr_pos, global_fit_cap, replace=False)
    gmodel.fit(Xall.iloc[tr_pos], y[tr_pos])
    pred = pd.Series(gmodel.predict_proba(Xall[te])[:,1], index=df.index[te])
    src = pd.Series("global", index=df.index[te])
    tr_idx, te_idx = df.index[tr], df.index[te]
    counts = df.loc[tr_idx].groupby("pitcher").size()
    eligible = counts[counts >= min_train].index
    for pid in eligible:
        ptr = tr_idx[df.loc[tr_idx,"pitcher"].eq(pid).to_numpy()]
        pte = te_idx[df.loc[te_idx,"pitcher"].eq(pid).to_numpy()]
        if len(pte) == 0 or df.loc[ptr,"is_csw"].nunique() < 2: continue
        m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
        m.fit(Xall.loc[ptr], y[df.index.get_indexer(ptr)])
        pred.loc[pte] = m.predict_proba(Xall.loc[pte])[:,1]; src.loc[pte] = "per_pitcher"
    y_te = df.loc[te_idx,"is_csw"].to_numpy()
    cov = float((src == "per_pitcher").mean())
    res = {"coverage_per_pitcher": round(cov,4), "fallback_global": round(1-cov,4),
           "n_eligible_pitchers": int(len(eligible)),
           "weighted_all": metrics(y_te, pred.to_numpy()),
           "per_pitcher_only": metrics(df.loc[src[src=="per_pitcher"].index,"is_csw"].to_numpy(),
                                       pred[src=="per_pitcher"].to_numpy()) if cov>0 else None,
           "global_fallback_only": metrics(df.loc[src[src=="global"].index,"is_csw"].to_numpy(),
                                            pred[src=="global"].to_numpy()) if cov<1 else None}
    return res, pred, src
