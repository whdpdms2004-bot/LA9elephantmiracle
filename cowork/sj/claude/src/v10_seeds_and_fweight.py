"""V10: 시드 확대(8->20) + R/F 가중치 2차원 그리드.

왜 이 둘인가
    지금까지 9라운드에서 성공/실패의 구분선이 명확해졌다.
        성공 = 모델이 못 보던 정보나 구조를 새로 넣은 것
               플래툰 +13.75 / OUTSIDE 분할 +2.64 / CatBoost 계열 +1.23
        실패 = 이미 있는 정보의 재표현
               성분별 플래툰 +0.16 / 스태킹 +0.37 / 감쇠외삽 0 / 유형 residual 0

    남은 건 (a) 팀 근거상 90% 이상 전이되는 범주(앙상블·배깅)와
            (b) 아직 한 번도 안 재본 자유도다.

    (a) 시드 8 -> 20. 팀 표준은 20~30인데 sj 성분 라인은 8이다.
        팀통합 1-5: 앙상블/배깅은 새 가정이 안 들어가서 전이가 안전하다.
    (b) F 행 가중치. 현재 F 는 손대지 않는다(w_F=0). P12 에서 w=0.20 을 전 행에
        적용하면 F 가 -3.54 로 손해였지만 0.05/0.10 은 측정한 적이 없다.
        F 는 Val2024 30,010행(11.8%)이고 BSS 528.8 로 R(834.1)보다 훨씬 나쁘다.

설계
    5성분 x 2계열(XGB, CatBoost) x 20시드 = 200 모델을 한 번 학습해두고,
    w_R x w_F 2차원 그리드를 사후 적용한다. 시드 8/12/16/20 곡선도 같이 낸다.

판정: Val2024 전체 BSS, 프로덕션 836.503 대비. 분모 SE_row.
      w 는 사전 등록 0.20 을 기준으로 삼고 argmax 는 참고만 본다.
출력: outputs/v10_seeds_fweight.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, make_priors,
                     add_stateless, encode, metrics)

PROD = (Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
        / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88,
         101, 202, 303, 404, 505, 606, 707, 808,
         111, 222, 333, 444]
SEED_CURVE = [8, 12, 16, 20]
WR = [0.15, 0.20, 0.25, 0.30]
WF = [0.00, 0.05, 0.10, 0.15, 0.20]
DROP = ["row_id", TARGET]
N_ROUNDS = 400
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


def build():
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
    d = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "y": y_all[tr]})
    lg = float(d["y"].mean()); K = 300
    ga = d.groupby("p")["y"].agg(["sum", "size"])
    gh = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    eb_p = (ga["sum"] + K * lg) / (ga["size"] + K)
    eb_ph = (gh["sum"] + K * lg) / (gh["size"] + K)
    key = pd.MultiIndex.from_arrays([pid, bhand])
    lv = np.where(np.isnan(eb_ph.reindex(key).to_numpy()), lg,
                  eb_ph.reindex(key).to_numpy())
    pv = np.where(np.isnan(pd.Series(pid).map(eb_p).to_numpy()), lg,
                  pd.Series(pid).map(eb_p).to_numpy())
    sz = gh["size"].reindex(key).fillna(0.0).to_numpy()
    sp, rel = lv - pv, sz / (sz + K)
    out["platoon_split"], out["platoon_split_rel"] = sp, rel
    out["platoon_split_w"] = sp * rel
    return out.to_numpy(np.float32)


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


X = build()
t0 = time.time()
per_seed = {}          # per_seed[tag] = (n_seed, n_val) 배열, XGB+Cat 평균
for tag, arr in LAB.items():
    m = tr & ~np.isnan(arr)
    rate = float(np.nanmean(arr[tr]))
    prm = {**BASE_PARAMS, "base_score": extrap(arr), **params_for(rate)}
    d_tr = xgb.DMatrix(X[m], label=arr[m]); d_va = xgb.DMatrix(X[va])
    p_tr = Pool(X[m], arr[m]); p_va = Pool(X[va])
    acc = np.empty((len(SEEDS), int(va.sum())))
    for i, s in enumerate(SEEDS):
        px = xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                       verbose_eval=False).predict(d_va)
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s, task_type="GPU", verbose=0)
        c.fit(p_tr)
        acc[i] = 0.5 * px + 0.5 * c.predict_proba(p_va)[:, 1]
    per_seed[tag] = acc
    print(f"  [{tag:>2}] rate {rate:.4f}  {len(SEEDS)}시드 x 2계열 완료 "
          f"{time.time()-t0:.0f}s", flush=True)

prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
y_va = y_all[va]
gt = prod["game_type"].astype(str).to_numpy()
is_r = gt == "R"
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
bm = metrics(y_va, p_prod, game_type=gt)
null = y_va.mean() * (1 - y_va.mean())


def compose(k):
    p = {t: np.clip(per_seed[t][:k].mean(axis=0), EPS, 1 - EPS) for t in LAB}
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


print("\n시드 개수 -> 성분 라인 단독 BSS / 결합 (w_R=0.20, w_F=0)")
rows = []
for k in SEED_CURVE:
    ie = compose(k)
    q = p_prod.copy(); q[is_r] = 0.20 * ie[is_r] + 0.80 * p_prod[is_r]
    mm = metrics(y_va, np.clip(q, EPS, 1 - EPS), game_type=gt)
    print(f"  {k:>2}시드  단독 {metrics(y_va, ie)['bss_raw']:7.2f}  "
          f"결합 {mm['bss_raw']:8.3f}  ΔBSS {mm['bss_raw']-bm['bss_raw']:+7.3f}",
          flush=True)

ie20 = compose(len(SEEDS))
print(f"\nw_R x w_F 2차원 그리드 (20시드)  프로덕션 {bm['bss_raw']:.3f} 대비 ΔBSS")
print(f"{'w_R\\w_F':>8}" + "".join(f"{v:>9.2f}" for v in WF))
for wr in WR:
    line = f"{wr:>8.2f}"
    for wf in WF:
        q = p_prod.copy()
        q[is_r] = wr * ie20[is_r] + (1 - wr) * p_prod[is_r]
        q[~is_r] = wf * ie20[~is_r] + (1 - wf) * p_prod[~is_r]
        q = np.clip(q, EPS, 1 - EPS)
        mm = metrics(y_va, q, game_type=gt)
        d = mm["bss_raw"] - bm["bss_raw"]
        dr = (p_prod - y_va) ** 2 - (q - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"n_seed": len(SEEDS), "w_r": wr, "w_f": wf,
                     "bss": mm["bss_raw"], "dbss": d, "se_row": se, "t_row": d / se,
                     "r_bss": mm["r_bss"], "f_bss": mm["f_bss"],
                     "pred_mean": mm["pred_mean"]})
        line += f"{d:>+9.2f}"
    print(line, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v10_seeds_fweight.csv", index=False)
ref = res[(res.w_r == 0.20) & (res.w_f == 0.0)].iloc[0]
best = res.sort_values("dbss", ascending=False).iloc[0]
print(f"\n사전등록 (w_R=0.20, w_F=0)  ΔBSS {ref.dbss:+.3f}  t_row {ref.t_row:+.2f}"
      f"   [submit_025 8시드는 +17.31]")
print(f"최고 (w_R={best.w_r:.2f}, w_F={best.w_f:.2f})  ΔBSS {best.dbss:+.3f}  "
      f"t_row {best.t_row:+.2f}   F BSS {best.f_bss:.2f} (기준 {bm['f_bss']:.2f})")
print(f"\nsaved -> {OUT/'v10_seeds_fweight.csv'}")
