"""V5: sj 라인 x hw v8 라인 2자 결합 — 팀 결합 가설을 기다리지 않고 검증한다.

배경
    1000 도달의 유일한 현실적 경로는 팀 5인 결합이다(예상 1017~1056, plan.md 1순위).
    그런데 팀원 Val2024 예측을 기다리는 중이다. hw v8 은 내가 직접 실행했으므로
    모델과 프로토콜(fit<2024, val==2024)이 손에 있다. 지금 2자 결합을 재면
    팀 결합 가설의 핵심 가정(상관이 낮아 결합 이득이 크다)을 즉시 검증할 수 있다.

    hw v8 Phase1 635.88 vs sj 850. 약한 라인이지만 plan.md 5-5 원칙대로
    "약한 라인을 억지로 빼지 않는다 — w* 가 알아서 작은 가중치를 준다."

가중치 산출 (plan.md / 팀통합 5-3 의 규정 준수 대안)
    리더보드 역산이 아니라 홀드아웃 2024 실제 라벨로 닫힌 형태를 푼다.
        D = [p_i - r]        r = 학습 데이터 기반 상수
        M = D^T D / n,  A = D^T (y - r) / n
        w = M^-1 A
    리더보드를 한 번도 경유하지 않으므로 09.11 검증에서 방어 가능하다.

부산물
    cowork/hw/val2024_pred.csv   hw 라인의 공통 Val2024 예측 (16번 문서 규격)
    cowork/sj/val2024_pred.csv   sj 라인의 공통 Val2024 예측
    submission_v8/extra_seeds_by_sj.json  CatBoost 시드 5->20 곡선

출력: outputs/v5_two_line_ensemble.csv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]          # cowork/sj
REPO = ROOT.parent.parent                            # 저장소 루트
HW = REPO / "cowork" / "hw"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(HW))
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, make_priors,
                     add_stateless, encode, metrics)
import train_best_model_v8 as V8

PROD = (ROOT / "experiment" / "model_optimization" / "pitcher_cluster_matchup"
        / "reports" / "reverse20_submission_oof.parquet")
NPZ = PROD.parent / "reverse20_submission_components.npz"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]
N_ROUNDS, W_IE = 400, 0.20
EXTRA_CAT = list(range(2031, 2046))
CURVE = [5, 8, 10, 12, 15, 20]
EPS = 1e-7
t0 = time.time()

RATES = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
         "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
         "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
         "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
PREV_S = [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
PREV_M = [f"asof_pitcher_prev{k}_game_middle_rate" for k in (1, 3, 5)]


def logit(p):
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


# ================================================================= sj 라인
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


def platoon(K=300):
    d = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "y": y_all[tr]})
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


def build_sj():
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


def extrap(a):
    m = tr & ~np.isnan(a)
    s = pd.Series(a[m]).groupby(pd.Series(season[m])).mean().sort_index()
    last = float(s.iloc[-1])
    return float(np.clip(last + (last - float(s.iloc[0]))
                         / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))


X = build_sj()
comp = {}
for tag, arr in LAB.items():
    m = tr & ~np.isnan(arr)
    d_tr = xgb.DMatrix(X[m], label=arr[m]); d_va = xgb.DMatrix(X[va])
    acc = np.zeros(int(va.sum()))
    for s in SEEDS:
        acc += xgb.train({**BASE_PARAMS, "base_score": extrap(arr), "seed": s},
                         d_tr, num_boost_round=N_ROUNDS,
                         verbose_eval=False).predict(d_va)
    comp[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
p_ie = np.clip(1 - (comp["m"] + comp["r"] - comp["mr"] + comp["o"]), EPS, 1 - EPS)
V8.log("sj 성분 라인 완료", t0)

prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
z = np.load(NPZ, allow_pickle=True)
y_va = y_all[va]
gt = prod["game_type"].astype(str).to_numpy()
is_r = gt == "R"
p021 = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
p022 = np.clip(p021 + 0.075 * 0.6085 * z["reverse20"].astype(np.float64), EPS, 1 - EPS)
p_sj = p022.copy()
p_sj[is_r] = W_IE * p_ie[is_r] + (1 - W_IE) * p022[is_r]     # = submit_024 구성
print(f"  sj submit_022 {metrics(y_va,p022)['bss_raw']:.3f}  "
      f"submit_024 {metrics(y_va,p_sj)['bss_raw']:.3f}", flush=True)

# ================================================================= hw 라인
raw = pd.read_csv(V8.DATA_DIR / "train.csv")
test_cols = pd.read_csv(V8.DATA_DIR / "test.csv", nrows=0).columns.tolist()
full = V8.add_trend(raw)
fit_h = full[full.season < 2024].copy()
val_h = full[full.season == 2024].copy()
y_fit = fit_h[V8.TARGET].to_numpy()
assert (val_h[V8.TARGET].to_numpy() == y_va).all(), "행 정렬 불일치"
num_cols = [c for c in test_cols if c not in ([V8.ID] + V8.BASELINE_CATS)]
num_cols += [c for c in full.columns
             if c not in test_cols and c != V8.TARGET and c not in num_cols]
med = fit_h[num_cols].median()
cmaps = V8.build_cat_maps(fit_h, V8.BASELINE_CATS)
cols_h = V8.BASELINE_CATS + num_cols
xf_cb = V8.matrix_catboost(fit_h, cols_h, num_cols, med)
xv_cb = V8.matrix_catboost(val_h, cols_h, num_cols, med)
xf_lg = V8.matrix_lightgbm(fit_h, cols_h, num_cols, med, cmaps)
xv_lg = V8.matrix_lightgbm(val_h, cols_h, num_cols, med, cmaps)
V8.log("hw 행렬 준비", t0)

cat_p, cat_b = {}, {}
for sd in V8.CAT_SEEDS + EXTRA_CAT:
    mdl, bi = V8.train_cb_member(sd, xf_cb, y_fit, xv_cb, y_va)
    cat_p[sd] = mdl.predict_proba(xv_cb)[:, 1]
    cat_b[sd] = V8.score(y_va, cat_p[sd])[1]
    V8.log(f"  Cat seed={sd} iter={bi} BSS={cat_b[sd]:7.2f}", t0)
lgb_p, lgb_b = {}, {}
for sd in V8.LGB_SEEDS:
    mdl, bi = V8.train_lgb_member(sd, xf_lg, y_fit, xv_lg, y_va)
    lgb_p[sd] = mdl.predict(xv_lg, num_iteration=mdl.best_iteration)
    lgb_b[sd] = V8.score(y_va, lgb_p[sd])[1]
lgb_mean = np.mean([lgb_p[s] for s in V8.LGB_SEEDS], axis=0)

curve = []
allc = V8.CAT_SEEDS + EXTRA_CAT
for k in CURVE:
    cb = np.mean([cat_p[s] for s in allc[:k]], axis=0)
    ens = 0.5 * cb + 0.5 * lgb_mean
    curve.append({"cat_seeds": k, "cat_ens": V8.score(y_va, cb)[1],
                  "final": V8.score(y_va, ens)[1],
                  "delta_anchor": V8.score(y_va, ens)[1] - V8.KNOWN_ANCHOR_BSS_2024})
p_hw = np.clip(0.5 * np.mean([cat_p[s] for s in allc], axis=0) + 0.5 * lgb_mean,
               EPS, 1 - EPS)
json.dump({"curve": curve,
           "cat_individual": {"n": len(allc),
                              "mean": float(np.mean(list(cat_b.values()))),
                              "sd": float(np.std(list(cat_b.values()), ddof=1))},
           "note": "하이퍼파라미터 불변, CatBoost 시드 개수만 변화"},
          open(V8.OUT_DIR / "extra_seeds_by_sj.json", "w"), indent=1, ensure_ascii=False)

print("\n" + "=" * 84)
print("CatBoost 시드 개수 -> hw v8 Phase1 앙상블 BSS (하이퍼파라미터 불변)")
print("=" * 84)
for c in curve:
    print(f"  cat시드 {c['cat_seeds']:>2}  Cat앙상블 {c['cat_ens']:8.2f}  "
          f"최종 {c['final']:8.2f}  anchor(650.54) 대비 {c['delta_anchor']:+8.2f}")

# ============================================================ 2자 결합
print("\n" + "=" * 84)
print("sj x hw 2자 결합")
print("=" * 84)
m_sj, m_hw = metrics(y_va, p_sj), metrics(y_va, p_hw)
corr_p = float(np.corrcoef(p_sj, p_hw)[0, 1])
corr_l = float(np.corrcoef(logit(p_sj), logit(p_hw))[0, 1])
print(f"  sj  {m_sj['bss_raw']:8.3f}   hw  {m_hw['bss_raw']:8.3f}")
print(f"  상관  확률 {corr_p:.4f}   로짓 {corr_l:.4f}   "
      f"({'0.85 이하 - 결합 이득 큼' if corr_l <= 0.85 else '0.9 이상 - 이득 작음'})")

r = float(y_all[tr].mean())
D = np.column_stack([p_sj - r, p_hw - r])
M = D.T @ D / len(D)
A = D.T @ (y_va - r) / len(D)
w = np.linalg.solve(M, A)
print(f"  w* = M^-1 A  ->  sj {w[0]:.4f}  hw {w[1]:.4f}  (합 {w.sum():.4f})")

rows = []
null = y_va.mean() * (1 - y_va.mean())
for name, p in [("sj_only", p_sj), ("hw_only", p_hw),
                ("optimal_w", np.clip(r + D @ w, EPS, 1 - EPS))] + \
               [(f"grid_hw{v:.2f}", np.clip((1 - v) * p_sj + v * p_hw, EPS, 1 - EPS))
                for v in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]]:
    mm = metrics(y_va, p, game_type=gt)
    d = mm["bss_raw"] - m_sj["bss_raw"]
    dr = (p_sj - y_va) ** 2 - (p - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"variant": name, "bss": mm["bss_raw"], "dbss_vs_sj": d,
                 "se_row": se, "t_row": d / se if se > 0 else np.nan,
                 "pred_mean": mm["pred_mean"]})
    print(f"  {name:<14}{mm['bss_raw']:>10.3f}{d:>+9.3f}  t_row {d/se if se>0 else 0:+6.2f}")

pd.DataFrame(rows).to_csv(OUT / "v5_two_line_ensemble.csv", index=False)
ids = df.loc[va, "row_id"].to_numpy()
pd.DataFrame({"row_id": ids, "control_success": p_hw}).to_csv(
    HW / "val2024_pred.csv", index=False)
pd.DataFrame({"row_id": ids, "control_success": p_sj}).to_csv(
    ROOT / "val2024_pred.csv", index=False)
print(f"\nsaved -> {OUT/'v5_two_line_ensemble.csv'}")
print(f"        {HW/'val2024_pred.csv'}  (hw 라인, 16번 문서 규격)")
print(f"        {ROOT/'val2024_pred.csv'}  (sj 라인)")
