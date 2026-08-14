"""V6: 성분 분해를 더 민다 — 성분별 개별 튜닝 + OUTSIDE 분할 + 5성분 확장.

근거 (V3 실측)
    같은 피처 / 같은 계열인데
        성분 분해 있음  ie_CF_all   +14.58   corr 0.8625
        성분 분해 없음  dir_CF_all   +7.08   corr 0.8844
    분해가 이득의 절반 이상을 만들고 상관도 더 낮다. 여기가 가장 유력한 축이다.

지금까지 안 한 것 세 가지

  1. 성분별 개별 파라미터
     네 성분의 기저율이 10배 차이인데 파라미터를 공유하고 있다.
        m 0.1866   r 0.2613   o 0.1161   mr 0.0344
     희소한 mr 은 더 강한 규제와 더 낮은 base_score 가 맞을 수 있다.

  2. OUTSIDE 분할
     OUTSIDE(194,093) = 실패 and not middle and not reverse 인 잔여 버킷인데
     ball=1 (159,871) 과 ball=0 (34,222) 이 섞여 있다. 후자는 존 안인데 실패한
     '문제 정의에 없는 네번째 모드'다. 성격이 다른 둘을 한 모델이 맞추고 있다.
        P(success) = 1 - [p_m + p_r - p_mr + p_ob + p_oz]
     ob = OUTSIDE and ball,  oz = OUTSIDE and not ball. 둘은 정의상 배타다.

  3. 교집합 항 확장
     현재 포함배제에서 p_mr 만 모델링한다. OUTSIDE 는 m/r 과 배타라 교집합이
     없지만, 분할 후 ob/oz 도 서로 배타이므로 추가 항은 필요 없다.
     대신 mr 이 희소(0.0344)해서 추정이 불안정할 수 있으니 스케일을 따로 둔다.

판정: Val2024 전체 BSS, 프로덕션 836.503 대비. 고정 w 로 비교하고 argmax 는 참고만.
출력: outputs/v6_component_deepen.csv
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
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]
N_ROUNDS = 400
WS = [0.15, 0.20, 0.25, 0.30, 0.35]
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
LAB4 = {"m": ym, "r": yr, "o": yo,
        "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan)}
LAB5 = {"m": ym, "r": yr,
        "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
        "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan),
        "mr": LAB4["mr"]}
for k, v in list(LAB4.items()) + list(LAB5.items()):
    print(f"  {k:>3} 기저율 {np.nanmean(v[tr]):.6f}  n={int(np.nansum(v[tr])):,}", flush=True)


def platoon(K=300):
    d = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "y": y_all[tr]})
    lg = float(d["y"].mean())
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
    return lv - pv, sz / (sz + K)


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
    sp, rel = platoon()
    out["platoon_split"], out["platoon_split_rel"] = sp, rel
    out["platoon_split_w"] = sp * rel
    return out.to_numpy(np.float32)


X = build()


def extrap(a):
    m = tr & ~np.isnan(a)
    s = pd.Series(a[m]).groupby(pd.Series(season[m])).mean().sort_index()
    last = float(s.iloc[-1])
    return float(np.clip(last + (last - float(s.iloc[0]))
                         / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))


# 기저율에 맞춘 성분별 파라미터. 희소할수록 얕고 강하게 규제한다.
def params_for(rate):
    if rate < 0.06:                      # mr, oz 처럼 희소
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:                      # o, ob
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def fit_xgb(arr, extra=None):
    m = tr & ~np.isnan(arr)
    prm = {**BASE_PARAMS, "base_score": extrap(arr), **(extra or {})}
    d_tr = xgb.DMatrix(X[m], label=arr[m]); d_va = xgb.DMatrix(X[va])
    acc = np.zeros(int(va.sum()))
    for s in SEEDS:
        acc += xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                         verbose_eval=False).predict(d_va)
    return np.clip(acc / len(SEEDS), EPS, 1 - EPS)


def fit_cat(arr):
    m = tr & ~np.isnan(arr)
    p_tr, p_va = Pool(X[m], arr[m]), Pool(X[va])
    acc = np.zeros(int(va.sum()))
    for s in SEEDS:
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s, task_type="GPU", verbose=0)
        c.fit(p_tr)
        acc += c.predict_proba(p_va)[:, 1]
    return np.clip(acc / len(SEEDS), EPS, 1 - EPS)


t0 = time.time()
arms = {}
for tag, fn in [("xgb", fit_xgb), ("cat", fit_cat)]:
    p4 = {k: (fn(v) if fn is fit_cat else fn(v)) for k, v in LAB4.items()}
    arms[f"A4_shared_{tag}"] = np.clip(
        1 - (p4["m"] + p4["r"] - p4["mr"] + p4["o"]), EPS, 1 - EPS)
    print(f"  [{tag}] 4성분 공유파라미터 완료 {time.time()-t0:.0f}s", flush=True)

p4t = {k: fit_xgb(v, params_for(float(np.nanmean(v[tr])))) for k, v in LAB4.items()}
arms["B4_tuned_xgb"] = np.clip(1 - (p4t["m"] + p4t["r"] - p4t["mr"] + p4t["o"]),
                               EPS, 1 - EPS)
print(f"  4성분 개별튜닝 완료 {time.time()-t0:.0f}s", flush=True)

p5 = {k: fit_xgb(v, params_for(float(np.nanmean(v[tr])))) for k, v in LAB5.items()}
arms["C5_split_xgb"] = np.clip(
    1 - (p5["m"] + p5["r"] - p5["mr"] + p5["ob"] + p5["oz"]), EPS, 1 - EPS)
print(f"  5성분 분할 완료 {time.time()-t0:.0f}s", flush=True)

p5c = {k: fit_cat(v) for k, v in LAB5.items()}
arms["D5_split_cat"] = np.clip(
    1 - (p5c["m"] + p5c["r"] - p5c["mr"] + p5c["ob"] + p5c["oz"]), EPS, 1 - EPS)
arms["E5_split_xc"] = np.clip(
    1 - sum(0.5 * (p5[k] + p5c[k]) * (-1 if k == "mr" else 1) for k in LAB5),
    EPS, 1 - EPS)
print(f"  5성분 CatBoost/혼합 완료 {time.time()-t0:.0f}s", flush=True)

prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
y_va = y_all[va]
gt = prod["game_type"].astype(str).to_numpy()
is_r = gt == "R"
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
bm = metrics(y_va, p_prod, game_type=gt)
null = y_va.mean() * (1 - y_va.mean())

rows = []
print(f"\n{'arm':<18}{'단독BSS':>10}{'corr':>8}   " + "".join(f"w{w:<6.2f}" for w in WS))
for name, p_c in arms.items():
    solo = metrics(y_va, p_c)["bss_raw"]
    corr = float(np.corrcoef(logit(p_prod), logit(p_c))[0, 1])
    line = f"{name:<18}{solo:>10.2f}{corr:>8.4f}   "
    for w in WS:
        q = p_prod.copy()
        q[is_r] = w * p_c[is_r] + (1 - w) * p_prod[is_r]
        q = np.clip(q, EPS, 1 - EPS)
        mm = metrics(y_va, q, game_type=gt)
        d = mm["bss_raw"] - bm["bss_raw"]
        dr = (p_prod - y_va) ** 2 - (q - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"arm": name, "solo_bss": solo, "corr": corr, "w": w,
                     "bss": mm["bss_raw"], "dbss": d, "se_row": se, "t_row": d / se})
        line += f"{d:+7.2f}"
    print(line, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v6_component_deepen.csv", index=False)
ref = res[(res.arm == "A4_shared_xgb") & (res.w == 0.20)]["dbss"].iloc[0]
best = res[res.w == 0.20].sort_values("dbss", ascending=False).iloc[0]
print(f"\n고정 w=0.20 기준")
print(f"  현행(A4_shared_xgb)  {ref:+.3f}")
print(f"  최고 {best.arm:<18} {best.dbss:+.3f}   차이 {best.dbss-ref:+.3f}  "
      f"t_row {best.t_row:+.2f}")
print(f"\nsaved -> {OUT/'v6_component_deepen.csv'}")
