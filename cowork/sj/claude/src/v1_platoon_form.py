"""V1: 플래툰 피처의 형태(form) 탐색 — 레벨 vs 스플릿, 정적 vs as-of.

예나 커밋 7ebfb3e 반영
  - 챔피언 피처는 hand_success = asof_pitcher_vs_hand_success_rate_smoothed
    즉 '레벨'(K=20)이다. sj 라인에는 split(차이)만 있고 레벨이 없었다.
  - 예나는 split 을 행 단위 as-of(자기 행 제외)로 만든다. sj 는 정적 테이블이다.

중요한 구분
  test.csv 에는 라벨이 없으므로 추론 시점에 as-of 를 계산할 수 없다.
  따라서 검증/배포 행은 반드시 학습 데이터로 만든 '동결 테이블' 값을 받는다.
  쟁점은 '학습 행을 어떻게 featurize 하는가' 하나뿐이다.

    S static   학습 행도 동결 테이블   -> 자기 라벨 포함(누수), 배포와 형태 일치
    A row-asof 학습 행은 직전까지 누적 -> 누수 없음, 배포와 형태 불일치
    B season   학습 행은 이전 시즌만   -> 누수 없음, 배포와 구조 일치

검증 행은 세 방식 모두 동결 테이블로 통일한다. 그래야 정직한 비교다.

피처 형태
    level  EB(투수 x 타자손 성공률)                  <- 예나 챔피언 (K=20)
    split  EB(투수 x 타자손) - EB(투수 전체)          <- 팀통합 표준 (K=200~300)
    both   둘 다

판정: Val2024 전체 BSS, 프로덕션 submit_021(836.503) 대비. 분모 SE_row.
출력: outputs/v1_platoon_form.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, make_priors,
                     add_stateless, encode, metrics)

PROD = (Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
        / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]
N_ROUNDS = 400
WS = [0.10, 0.15, 0.20, 0.25, 0.30]
K_LEVEL, K_SPLIT = 20, 300
EPS = 1e-7

RATES = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
         "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
         "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
         "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
PREV_S = [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
PREV_M = [f"asof_pitcher_prev{k}_game_middle_rate" for k in (1, 3, 5)]


def logit(p):
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
asof_n = df["asof_pitcher_n"].to_numpy(np.float64)
pid = df["pitcher_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
ok = df["label_ok"].to_numpy() == 1
tr, va = season < 2024, season == 2024

LAB = {"succ": y_all,
       "m": np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan),
       "r": np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan),
       "o": np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)}
LAB["mr"] = np.where(ok, (LAB["m"] == 1) & (LAB["r"] == 1), np.nan)


# ------------------------------------------------------- 플래툰 3방식
def frozen_table(mask, K):
    """학습 데이터 전체로 만든 동결 EB. 검증/배포 행은 항상 이걸 받는다."""
    d = pd.DataFrame({"p": pid[mask], "h": bhand[mask], "y": y_all[mask]})
    league = float(d["y"].mean())
    ga = d.groupby("p")["y"].agg(["sum", "size"])
    gh = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    eb_p = (ga["sum"] + K * league) / (ga["size"] + K)
    eb_ph = (gh["sum"] + K * league) / (gh["size"] + K)
    return eb_ph, eb_p, league, gh["size"]


def apply_frozen(eb_ph, eb_p, league, size_ph, K):
    key = pd.MultiIndex.from_arrays([pid, bhand])
    lv = eb_ph.reindex(key).to_numpy()
    pv = pd.Series(pid).map(eb_p).to_numpy()
    sz = size_ph.reindex(key).fillna(0.0).to_numpy()
    lv = np.where(np.isnan(lv), league, lv)
    pv = np.where(np.isnan(pv), league, pv)
    return lv, lv - pv, sz / (sz + K)


def row_asof(K, league):
    """행 단위 as-of — 자기 행 제외한 직전까지 누적. CSV 순서를 시간 순서로 본다."""
    d = pd.DataFrame({"p": pid, "h": bhand, "y": y_all})
    gph = d.groupby(["p", "h"])["y"]
    n_ph = gph.cumcount().to_numpy(np.float64)
    s_ph = (gph.cumsum().to_numpy(np.float64) - y_all)
    gp = d.groupby("p")["y"]
    n_p = gp.cumcount().to_numpy(np.float64)
    s_p = (gp.cumsum().to_numpy(np.float64) - y_all)
    eb_ph = (s_ph + K * league) / (n_ph + K)
    eb_p = (s_p + K * league) / (n_p + K)
    return eb_ph, eb_ph - eb_p, n_ph / (n_ph + K)


def season_asof(K, league):
    """시즌 단위 as-of — 이전 시즌만. 배포(이전 시즌 전부)와 구조가 같다."""
    d = pd.DataFrame({"p": pid, "h": bhand, "s": season, "y": y_all})
    agg_ph = d.groupby(["p", "h", "s"])["y"].agg(["sum", "size"]).sort_index()
    cum_ph = agg_ph.groupby(level=[0, 1]).cumsum() - agg_ph
    agg_p = d.groupby(["p", "s"])["y"].agg(["sum", "size"]).sort_index()
    cum_p = agg_p.groupby(level=0).cumsum() - agg_p
    kph = pd.MultiIndex.from_arrays([pid, bhand, season])
    kp = pd.MultiIndex.from_arrays([pid, season])
    s_ph = cum_ph["sum"].reindex(kph).fillna(0.0).to_numpy()
    n_ph = cum_ph["size"].reindex(kph).fillna(0.0).to_numpy()
    s_p = cum_p["sum"].reindex(kp).fillna(0.0).to_numpy()
    n_p = cum_p["size"].reindex(kp).fillna(0.0).to_numpy()
    eb_ph = (s_ph + K * league) / (n_ph + K)
    eb_p = (s_p + K * league) / (n_p + K)
    return eb_ph, eb_ph - eb_p, n_ph / (n_ph + K)


def platoon_columns(mode, use_level, use_split):
    """mode in {none, static, rowasof, seasonasof}. 검증 행은 항상 동결값."""
    if mode == "none":
        return {}
    out = {}
    for K, want in [(K_LEVEL, use_level), (K_SPLIT, use_split)]:
        if not want:
            continue
        eb_ph, eb_p, league, size_ph = frozen_table(tr, K)
        lv_f, sp_f, rel_f = apply_frozen(eb_ph, eb_p, league, size_ph, K)
        if mode == "static":
            lv, sp, rel = lv_f, sp_f, rel_f
        else:
            lv, sp, rel = (row_asof(K, league) if mode == "rowasof"
                           else season_asof(K, league))
            lv, sp, rel = lv.copy(), sp.copy(), rel.copy()
            lv[va], sp[va], rel[va] = lv_f[va], sp_f[va], rel_f[va]   # 검증은 동결
        tag = "lv" if K == K_LEVEL else "sp"
        if want and K == K_LEVEL:
            out[f"platoon_{tag}_level"] = lv
            out[f"platoon_{tag}_rel"] = rel
        if want and K == K_SPLIT:
            out[f"platoon_{tag}_split"] = sp
            out[f"platoon_{tag}_rel"] = rel
            out[f"platoon_{tag}_split_w"] = sp * rel
    return out


def build_features(platoon_cols):
    priors = make_priors(df.loc[tr])
    base = encode(add_stateless(df, priors))
    cols = [c for c in base.columns if c not in DROP and not c.startswith("y_")
            and c != "label_ok"]
    out = base[cols].copy()
    n = asof_n
    for c in RATES:
        pr = float(df.loc[tr, c].median())
        r = np.where(np.isnan(df[c].to_numpy(np.float64)), pr, df[c].to_numpy(np.float64))
        out[f"prof200_{c}"] = (n * r + 200 * pr) / (n + 200)
    ps = {c: np.where(np.isnan(df[c].to_numpy(np.float64)),
                      float(df.loc[tr, c].median()),
                      df[c].to_numpy(np.float64)) for c in PREV_S + PREV_M}
    out["prev_trend_s"] = ps[PREV_S[0]] - ps[PREV_S[2]]
    out["prev_trend_m"] = ps[PREV_M[0]] - ps[PREV_M[2]]
    out["prev_std_s"] = np.std(np.vstack([ps[c] for c in PREV_S]), axis=0)
    out["prev_std_m"] = np.std(np.vstack([ps[c] for c in PREV_M]), axis=0)
    out["prev_miss_cnt"] = sum(np.isnan(df[c].to_numpy(np.float64)).astype(np.float64)
                               for c in PREV_S + PREV_M)
    for k, (cs, cm) in enumerate(zip(PREV_S, PREV_M)):
        out[f"faildir_{k}"] = ps[cm] - (1 - ps[cs])
    out["rel200"] = n / (n + 200.0)
    for k, v in platoon_cols.items():
        out[k] = v
    return out.to_numpy(np.float32), list(out.columns)


def extrap(arr):
    m = tr & ~np.isnan(arr)
    s = pd.Series(arr[m]).groupby(pd.Series(season[m])).mean().sort_index()
    last = float(s.iloc[-1])
    slope = (last - float(s.iloc[0])) / (float(s.index[-1]) - float(s.index[0]))
    return float(np.clip(last + slope, 0.005, 0.995))


def bag(X, arr, bs):
    m = tr & ~np.isnan(arr)
    d_tr = xgb.DMatrix(X[m], label=arr[m])
    d_va = xgb.DMatrix(X[va])
    acc = np.zeros(int(va.sum()))
    for s in SEEDS:
        acc += xgb.train({**BASE_PARAMS, "base_score": bs, "seed": s}, d_tr,
                         num_boost_round=N_ROUNDS, verbose_eval=False).predict(d_va)
    return np.clip(acc / len(SEEDS), EPS, 1 - EPS)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
y_va = y_all[va]
gt = prod["game_type"].astype(str).to_numpy()
is_r = gt == "R"
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
bm = metrics(y_va, p_prod, game_type=gt)
null = y_va.mean() * (1 - y_va.mean())
print(f"프로덕션 base Val2024 {bm['bss_raw']:.3f}\n", flush=True)

ARMS = [
    ("V0_none", "none", False, False),
    ("V1_static_split", "static", False, True),
    ("V2_rowasof_split", "rowasof", False, True),
    ("V3_seasonasof_split", "seasonasof", False, True),
    ("V4_static_level", "static", True, False),
    ("V5_seasonasof_level", "seasonasof", True, False),
    ("V6_static_both", "static", True, True),
    ("V7_seasonasof_both", "seasonasof", True, True),
    ("V8_rowasof_both", "rowasof", True, True),
]

rows = []
for name, mode, lv, sp in ARMS:
    X, cols = build_features(platoon_columns(mode, lv, sp))
    comp = {t: bag(X, a, extrap(a)) for t, a in LAB.items()}
    p_dir = comp["succ"]
    p_ie = np.clip(1 - (comp["m"] + comp["r"] - comp["mr"] + comp["o"]), EPS, 1 - EPS)
    m_dir, m_ie = metrics(y_va, p_dir), metrics(y_va, p_ie)
    c_ie = float(np.corrcoef(logit(p_prod), logit(p_ie))[0, 1])
    print(f"[{name:<20}] feats {len(cols):>3}  direct {m_dir['bss_raw']:7.2f}  "
          f"p_ie {m_ie['bss_raw']:7.2f}  corr {c_ie:.4f}", flush=True)
    best = None
    for w in WS:
        q = p_prod.copy()
        q[is_r] = w * p_ie[is_r] + (1 - w) * p_prod[is_r]
        q = np.clip(q, EPS, 1 - EPS)
        mm = metrics(y_va, q, game_type=gt)
        d = mm["bss_raw"] - bm["bss_raw"]
        dr = (p_prod - y_va) ** 2 - (q - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"arm": name, "mode": mode, "level": lv, "split": sp, "w": w,
                     "n_features": len(cols), "direct_bss": m_dir["bss_raw"],
                     "ie_bss": m_ie["bss_raw"], "corr_prod_ie": c_ie,
                     "bss": mm["bss_raw"], "dbss": d, "se_row": se,
                     "t_row": d / se, "r_bss": mm["r_bss"], "f_bss": mm["f_bss"]})
        if best is None or d > best[1]:
            best = (w, d, d / se)
    print(f"{'':<22} 최고 w={best[0]:.2f}  ΔBSS {best[1]:+.3f}  t_row {best[2]:+.2f}",
          flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v1_platoon_form.csv", index=False)
print("\n" + "=" * 92)
print("★ Val2024 ΔBSS (프로덕션 836.503 대비)")
print("=" * 92)
print(res.pivot(index="arm", columns="w", values="dbss").round(2).to_string())
print("\n재료 품질")
q = res[res.w == 0.20][["arm", "n_features", "direct_bss", "ie_bss", "corr_prod_ie"]]
print(q.round(4).to_string(index=False))
print(f"\n현행 submit_024 = V1_static_split w=0.20 = "
      f"{res[(res.arm=='V1_static_split')&(res.w==0.20)]['dbss'].iloc[0]:+.3f}")
print(f"\nsaved -> {OUT/'v1_platoon_form.csv'}")
