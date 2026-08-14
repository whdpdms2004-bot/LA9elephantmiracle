"""V3: 성분 모델 계열 확장 — XGB 단일 -> XGB + LightGBM + CatBoost.

근거
  팀통합 2-3: "알고리즘 선택에 시간 쓰지 말고 서로 다른 걸 여러 개 만들어 섞어라"
  팀통합 2-2 (예나): LGB 단독 748.41 -> +Cat 50:50 899.76 (+151.35)
  팀통합 1-5: 앙상블/배깅은 전이율 90% 이상. 새 피처(40%)보다 안전하다
  현재 sj 성분 라인은 XGB 단일이다.

계열 가중은 개수가 아니라 계열 단위로 준다 (예나 T4).
  p_k = mean(p_k^xgb, p_k^lgb, p_k^cat)          <- 성분별로 먼저 평균
  p_ie = 1 - (p_m + p_r - p_mr + p_o)

두 순서를 비교한다.
  CF  compose-after-fuse : 성분별 계열평균 -> 합성   (분산 증폭이 작다)
  FC  fuse-after-compose : 계열별 합성 -> 평균

플래툰 형태는 V1 결과로 확정한 것을 쓴다 (PLATOON_MODE / USE_LEVEL / USE_SPLIT).

판정: Val2024 전체 BSS, 프로덕션 836.503 대비, 분모 SE_row.
출력: outputs/v3_family_expand.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, make_priors,
                     add_stateless, encode, metrics)

PROD = (Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
        / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]
N_ROUNDS = 400
WS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
EPS = 1e-7

# V1 결과로 확정 — 실행 전 반드시 갱신
PLATOON_MODE = "static"      # none / static / rowasof / seasonasof
USE_LEVEL, USE_SPLIT = True, True
K_LEVEL, K_SPLIT = 20, 300

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
COMPONENTS = ["m", "r", "o", "mr"]


def frozen_platoon(K):
    d = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "y": y_all[tr]})
    league = float(d["y"].mean())
    ga = d.groupby("p")["y"].agg(["sum", "size"])
    gh = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    eb_p = (ga["sum"] + K * league) / (ga["size"] + K)
    eb_ph = (gh["sum"] + K * league) / (gh["size"] + K)
    key = pd.MultiIndex.from_arrays([pid, bhand])
    lv = eb_ph.reindex(key).to_numpy()
    pv = pd.Series(pid).map(eb_p).to_numpy()
    sz = gh["size"].reindex(key).fillna(0.0).to_numpy()
    lv = np.where(np.isnan(lv), league, lv)
    pv = np.where(np.isnan(pv), league, pv)
    return lv, lv - pv, sz / (sz + K)


def build_features():
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
    if PLATOON_MODE != "none":
        if USE_LEVEL:
            lv, _, rel = frozen_platoon(K_LEVEL)
            out["platoon_level"] = lv
            out["platoon_level_rel"] = rel
        if USE_SPLIT:
            _, sp, rel = frozen_platoon(K_SPLIT)
            out["platoon_split"] = sp
            out["platoon_split_rel"] = rel
            out["platoon_split_w"] = sp * rel
    return out.to_numpy(np.float32), list(out.columns)


def extrap(arr):
    m = tr & ~np.isnan(arr)
    s = pd.Series(arr[m]).groupby(pd.Series(season[m])).mean().sort_index()
    last = float(s.iloc[-1])
    slope = (last - float(s.iloc[0])) / (float(s.index[-1]) - float(s.index[0]))
    return float(np.clip(last + slope, 0.005, 0.995))


X, COLS = build_features()
print(f"피처 {len(COLS)}개  플래툰 mode={PLATOON_MODE} level={USE_LEVEL} "
      f"split={USE_SPLIT}", flush=True)


def fit_xgb(arr, bs):
    m = tr & ~np.isnan(arr)
    d_tr = xgb.DMatrix(X[m], label=arr[m])
    d_va = xgb.DMatrix(X[va])
    acc = np.zeros(int(va.sum()))
    for s in SEEDS:
        acc += xgb.train({**BASE_PARAMS, "base_score": bs, "seed": s}, d_tr,
                         num_boost_round=N_ROUNDS, verbose_eval=False).predict(d_va)
    return acc / len(SEEDS)


def fit_lgb(arr, bs):
    m = tr & ~np.isnan(arr)
    init = float(np.log(bs / (1 - bs)))
    acc = np.zeros(int(va.sum()))
    for s in SEEDS:
        ds = lgb.Dataset(X[m], label=arr[m], init_score=np.full(int(m.sum()), init),
                         free_raw_data=False)
        b = lgb.train({"objective": "binary", "learning_rate": 0.03,
                       "num_leaves": 18, "min_data_in_leaf": 64,
                       "feature_fraction": 0.6, "bagging_fraction": 0.8,
                       "bagging_freq": 1, "lambda_l2": 2.0, "verbose": -1,
                       "num_threads": 24, "seed": s}, ds,
                      num_boost_round=N_ROUNDS)
        acc += 1.0 / (1.0 + np.exp(-(b.predict(X[va], raw_score=True) + init)))
    return acc / len(SEEDS)


def fit_cat(arr, bs):
    m = tr & ~np.isnan(arr)
    p_tr = Pool(X[m], arr[m])
    p_va = Pool(X[va])
    acc = np.zeros(int(va.sum()))
    for s in SEEDS:
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s, task_type="GPU", verbose=0)
        c.fit(p_tr)
        acc += c.predict_proba(p_va)[:, 1]
    return acc / len(SEEDS)


FAMILIES = {"xgb": fit_xgb, "lgb": fit_lgb, "cat": fit_cat}
preds = {}
for fam, fn in FAMILIES.items():
    t0 = time.perf_counter()
    preds[fam] = {}
    for tag, arr in LAB.items():
        preds[fam][tag] = np.clip(fn(arr, extrap(arr)), EPS, 1 - EPS)
    print(f"  [{fam}] 완료 {time.perf_counter()-t0:.0f}초  "
          f"succ BSS {metrics(y_all[va], preds[fam]['succ'])['bss_raw']:.2f}",
          flush=True)


def compose(p):
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["o"]), EPS, 1 - EPS)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
y_va = y_all[va]
gt = prod["game_type"].astype(str).to_numpy()
is_r = gt == "R"
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
bm = metrics(y_va, p_prod, game_type=gt)
null = y_va.mean() * (1 - y_va.mean())

cands = {}
for fam in FAMILIES:
    cands[f"ie_{fam}"] = compose(preds[fam])
cands["ie_CF_all"] = compose({t: np.mean([preds[f][t] for f in FAMILIES], axis=0)
                              for t in LAB})
cands["ie_FC_all"] = np.mean([compose(preds[f]) for f in FAMILIES], axis=0)
cands["ie_CF_xl"] = compose({t: np.mean([preds[f][t] for f in ("xgb", "lgb")], axis=0)
                             for t in LAB})
cands["ie_CF_xc"] = compose({t: np.mean([preds[f][t] for f in ("xgb", "cat")], axis=0)
                             for t in LAB})
cands["dir_CF_all"] = np.mean([preds[f]["succ"] for f in FAMILIES], axis=0)

rows = []
print(f"\n{'candidate':<14}{'단독BSS':>10}{'corr':>8}   " +
      "".join(f"w{w:<6.2f}" for w in WS), flush=True)
for name, p_c in cands.items():
    p_c = np.clip(p_c, EPS, 1 - EPS)
    solo = metrics(y_va, p_c)["bss_raw"]
    corr = float(np.corrcoef(logit(p_prod), logit(p_c))[0, 1])
    line = f"{name:<14}{solo:>10.2f}{corr:>8.4f}   "
    for w in WS:
        q = p_prod.copy()
        q[is_r] = w * p_c[is_r] + (1 - w) * p_prod[is_r]
        q = np.clip(q, EPS, 1 - EPS)
        mm = metrics(y_va, q, game_type=gt)
        d = mm["bss_raw"] - bm["bss_raw"]
        dr = (p_prod - y_va) ** 2 - (q - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"candidate": name, "solo_bss": solo, "corr_prod": corr, "w": w,
                     "bss": mm["bss_raw"], "dbss": d, "se_row": se, "t_row": d / se,
                     "r_bss": mm["r_bss"], "f_bss": mm["f_bss"]})
        line += f"{d:+7.2f}"
    print(line, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v3_family_expand.csv", index=False)
best = res.sort_values("dbss", ascending=False).iloc[0]
print(f"\n★ 최고: {best.candidate} w={best.w:.2f}  "
      f"Val2024 {best.bss:.3f}  ΔBSS {best.dbss:+.3f}  t_row {best.t_row:+.2f}")
print(f"   V1(submit_024) 기준 +13.752 대비 {best.dbss - 13.752:+.3f}")
print(f"\nsaved -> {OUT/'v3_family_expand.csv'}")
