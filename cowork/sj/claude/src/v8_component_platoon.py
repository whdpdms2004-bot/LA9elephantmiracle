"""V8: 성분별 플래툰 스플릿 — 좌우 신호를 성분 단위로 분리한다.

논리
    팀통합 1-1: asof_pitcher_success_rate 에 좌우 구분이 없어 15%p 가 사라진다.
                -> 투수별 플래툰 스플릿으로 복원 (sj Public 945 의 핵심)
    V6:         성분마다 기저율과 성격이 달라 따로 다뤄야 한다
                -> 성분별 파라미터 +0.91, OUTSIDE 분할 +2.64

    두 논리를 합치면: 플래툰도 성분별로 있어야 한다.
    현재는 control_success 기준 스플릿 하나로 다섯 성분을 다 커버하고 있다.

        split_k(p,h) = EB(투수 p, 타자손 h 의 성분 k 발생률)
                     - EB(투수 p 의 성분 k 발생률)

    REVERSE 는 정의상 '포수 요구 방향과 반대'라 좌우 매치업과 직결되지만
    MIDDLE(가운데 몰림)은 좌우와 무관할 수 있다. 하나로 뭉개면 서로 상쇄된다.

    E1a 실측이 뒷받침한다 — middle-reverse 의존이 pitcher_id 축에서 가장 강했고
    (246명 중 134명, lift 0.417~2.568) 현재 피처가 그걸 전혀 설명 못 했다
    (모델 잔차 상관 -0.0657).

방식
    V1 에서 정적 동결 테이블이 as-of 보다 나았으므로 같은 방식을 쓴다.
    5개 성분 x (split, rel) = 10 컬럼을 공유 행렬에 추가해 모든 성분 모델이
    서로의 좌우 신호도 볼 수 있게 한다.

    주효과 차감은 필수다 — 안 빼면 성분 기저율과 중복이고, V1 에서 정적 레벨이
    direct_bss 705.7 -> 187.5 로 붕괴한 것이 그 증거다.

판정: Val2024 전체 BSS, 프로덕션 836.503 대비, 고정 w=0.20 기준.
      계열은 XGB 단일로 비교한다 (CatBoost 이득 +1.23 은 가산적이라고 보고 분리).
출력: outputs/v8_component_platoon.csv
"""
import sys
import time
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
WS = [0.15, 0.20, 0.25, 0.30, 0.35]
K_GLOBAL = 300
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

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan),
       "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
       "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan)}


def eb_split(target, K):
    """정적 동결 테이블. 주효과 차감 필수. 학습 시즌만으로 만든다."""
    m = tr & ~np.isnan(target)
    d = pd.DataFrame({"p": pid[m], "h": bhand[m], "y": target[m]})
    lg = float(d["y"].mean())
    ga = d.groupby("p")["y"].agg(["sum", "size"])
    gh = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    eb_p = (ga["sum"] + K * lg) / (ga["size"] + K)
    eb_ph = (gh["sum"] + K * lg) / (gh["size"] + K)
    key = pd.MultiIndex.from_arrays([pid, bhand])
    lv = eb_ph.reindex(key).to_numpy()
    pv = pd.Series(pid).map(eb_p).to_numpy()
    sz = gh["size"].reindex(key).fillna(0.0).to_numpy()
    lv = np.where(np.isnan(lv), lg, lv)
    pv = np.where(np.isnan(pv), lg, pv)
    return lv - pv, sz / (sz + K)


def build(comp_platoon_K=None):
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
    sp, rel = eb_split(y_all, K_GLOBAL)                  # 전역 (control_success)
    out["platoon_split"], out["platoon_split_rel"] = sp, rel
    out["platoon_split_w"] = sp * rel
    if comp_platoon_K is not None:
        for tag, arr in LAB.items():
            s_k, r_k = eb_split(arr, comp_platoon_K)
            out[f"pl_{tag}_split"] = s_k
            out[f"pl_{tag}_rel"] = r_k
    return out.to_numpy(np.float32), list(out.columns)


def extrap(a):
    m = tr & ~np.isnan(a)
    s = pd.Series(a[m]).groupby(pd.Series(season[m])).mean().sort_index()
    last = float(s.iloc[-1])
    return float(np.clip(last + (last - float(s.iloc[0]))
                         / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def fit(X, arr):
    m = tr & ~np.isnan(arr)
    prm = {**BASE_PARAMS, "base_score": extrap(arr),
           **params_for(float(np.nanmean(arr[tr])))}
    d_tr = xgb.DMatrix(X[m], label=arr[m]); d_va = xgb.DMatrix(X[va])
    acc = np.zeros(int(va.sum()))
    for s in SEEDS:
        acc += xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                         verbose_eval=False).predict(d_va)
    return np.clip(acc / len(SEEDS), EPS, 1 - EPS)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
y_va = y_all[va]
gt = prod["game_type"].astype(str).to_numpy()
is_r = gt == "R"
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
bm = metrics(y_va, p_prod, game_type=gt)
null = y_va.mean() * (1 - y_va.mean())

# 성분별 좌우 스플릿의 크기를 먼저 본다 — 가설 확인
print("성분별 플래툰 스플릿 산포 (K=300, 학습 시즌)", flush=True)
for tag, arr in LAB.items():
    s_k, r_k = eb_split(arr, 300)
    hi = r_k > 0.5
    print(f"  {tag:>2}  기저율 {np.nanmean(arr[tr]):.4f}  "
          f"split sd {s_k[hi].std():.5f}  "
          f"p1~p99 {np.percentile(s_k[hi],1):+.4f}~{np.percentile(s_k[hi],99):+.4f}",
          flush=True)

t0 = time.time()
rows = []
print(f"\n{'arm':<20}{'피처':>5}{'단독BSS':>10}{'corr':>8}   "
      + "".join(f"w{w:<6.2f}" for w in WS), flush=True)
for name, K in [("F0_global_only", None), ("F1_comp_K150", 150),
                ("F2_comp_K300", 300), ("F3_comp_K600", 600)]:
    X, cols = build(K)
    p = {t: fit(X, a) for t, a in LAB.items()}
    p_ie = np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)
    solo = metrics(y_va, p_ie)["bss_raw"]
    corr = float(np.corrcoef(logit(p_prod), logit(p_ie))[0, 1])
    line = f"{name:<20}{len(cols):>5}{solo:>10.2f}{corr:>8.4f}   "
    for w in WS:
        q = p_prod.copy()
        q[is_r] = w * p_ie[is_r] + (1 - w) * p_prod[is_r]
        q = np.clip(q, EPS, 1 - EPS)
        mm = metrics(y_va, q, game_type=gt)
        d = mm["bss_raw"] - bm["bss_raw"]
        dr = (p_prod - y_va) ** 2 - (q - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"arm": name, "K": K, "n_features": len(cols), "solo_bss": solo,
                     "corr": corr, "w": w, "bss": mm["bss_raw"], "dbss": d,
                     "se_row": se, "t_row": d / se})
        line += f"{d:+7.2f}"
    print(line + f"   [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v8_component_platoon.csv", index=False)
ref = res[(res.arm == "F0_global_only") & (res.w == 0.20)]["dbss"].iloc[0]
best = res[res.w == 0.20].sort_values("dbss", ascending=False).iloc[0]
print(f"\n고정 w=0.20   기준선(F0) {ref:+.3f}   최고 {best.arm} {best.dbss:+.3f}  "
      f"차이 {best.dbss-ref:+.3f}  t_row {best.t_row:+.2f}")
print("  참고: submit_025(E5_split_xc, XGB+Cat)는 +17.31. "
      "여기는 XGB 단일이라 F0 가 +16.4 수준이어야 정상이다.")
print(f"\nsaved -> {OUT/'v8_component_platoon.csv'}")
