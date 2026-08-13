"""H) 워크로드/피로 피처의 순기여 격리 측정.
1) g_feats (워크로드 없음)  2) +워크로드  3) +워크로드 & CS/W 분해
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
import csw_pipeline as P

OUT = HERE / "out" / "v3"; C = HERE / "cache"
meta = json.loads((C / "meta_g.json").read_text())
df = pd.read_parquet(C / "features_g.parquet")
y = df["is_csw"].to_numpy(); yr = df["game_year"].to_numpy()
tr = np.isin(yr, P.TRAIN_YEARS); te = yr == 2019
trp, tep = np.where(tr)[0], np.where(te)[0]
rng = np.random.default_rng(0); N = 25000
cap = lambda i, n=N: i if len(i) <= n else rng.choice(i, n, replace=False)
bp = json.loads((OUT / "E_optuna.json").read_text())["best_params"]

g_feats = [c for c in meta["g_feats"] if c in df.columns]
wl = [c for c in meta["workload"] if c in df.columns]
h_feats = g_feats + [c for c in wl if c not in g_feats]

def run(cols, target="is_csw", mask=None):
    X = P.build_matrix(df, cols); t = df[target].to_numpy()
    idx = trp if mask is None else np.intersect1d(trp, np.where(mask)[0])
    m = LGBMClassifier(**bp, random_state=0, n_jobs=-1, verbose=-1)
    f = cap(idx); m.fit(X.iloc[f], t[f])
    return m, X, m.predict_proba(X.iloc[tep])[:, 1]

res, t0 = {}, time.time()
_, _, p_g = run(g_feats); res["1_no_workload"] = {**P.metrics(y[tep], p_g), "n_feats": len(g_feats)}
mH, Xh, p_h = run(h_feats); res["2_with_workload"] = {**P.metrics(y[tep], p_h), "n_feats": len(h_feats)}
res["delta_workload"] = round(res["2_with_workload"]["logloss"] - res["1_no_workload"]["logloss"], 5)
print(f"  1_no_workload   logloss {res['1_no_workload']['logloss']:.4f} auc {res['1_no_workload']['roc_auc']:.4f} ({len(g_feats)})")
print(f"  2_with_workload logloss {res['2_with_workload']['logloss']:.4f} auc {res['2_with_workload']['roc_auc']:.4f} ({len(h_feats)})")

sw = df["is_swing"].astype(bool).to_numpy()
mA, X, pA = run(h_feats, "is_swing")
mB, _, _ = run(h_feats, "is_whiff", sw)
mC, _, _ = run(h_feats, "is_called", ~sw)
Xte = X.iloc[tep]
pD = pA * mB.predict_proba(Xte)[:, 1] + (1 - pA) * mC.predict_proba(Xte)[:, 1]
res["3_workload_decomposed"] = P.metrics(y[tep], pD)
res["4_ensemble"] = P.metrics(y[tep], 0.5 * p_h + 0.5 * pD)
res["baselines"] = P.baselines(df, tr, te)
print(f"  3_decomposed    logloss {res['3_workload_decomposed']['logloss']:.4f} auc {res['3_workload_decomposed']['roc_auc']:.4f}")
print(f"  4_ensemble      logloss {res['4_ensemble']['logloss']:.4f} auc {res['4_ensemble']['roc_auc']:.4f}")

imp = pd.Series(mH.feature_importances_, index=Xh.columns).sort_values(ascending=False)
wcols = [c for c in imp.index if c in set(wl)]
res["workload_importance_share"] = round(float(imp[wcols].sum() / imp.sum()), 4)
res["workload_importance"] = {k: int(v) for k, v in imp[wcols].items()}
res["workload_rank_in_all"] = {c: int(list(imp.index).index(c)) + 1 for c in wcols}

# EDA 집계 저장 (노트북 그래프용)
def binmean(col, bins):
    b = pd.cut(df[col], bins)
    g = df.groupby(b, observed=True)["is_csw"].agg(["mean", "size"])
    return {str(k): {"csw": round(float(v["mean"]), 4), "n": int(v["size"])} for k, v in g.iterrows()}
res["eda_pitches_today"] = binmean("p_pitches_today", [0, 15, 30, 45, 60, 75, 90, 200])
res["eda_ratio"] = binmean("p_workload_ratio", [0, .25, .5, .75, 1.0, 1.25, 1.5, 10])
res["eda_z"] = binmean("p_workload_z", [-10, -1, 0, 1, 2, 10])
(OUT / "H_workload.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
print("워크로드 중요도 점유율:", res["workload_importance_share"], "| Δ:", res["delta_workload"], f"| {time.time()-t0:.1f}s")
