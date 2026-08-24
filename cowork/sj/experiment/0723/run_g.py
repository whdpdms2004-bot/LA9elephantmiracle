"""G) 타자 정보의 순기여 격리 측정 + 최고 모델 결합.

비교
  1) no_batter      : 타자 관련 피처 전부 제거 (73개 제거)
  2) batter_basic   : + 기존 타자 피처 (batter_te + 스카우팅 30)
  3) batter_full    : + 확장 타자 이력 (42개 추가) = g_feats
  4) batter_full + CS/W 분해 결합
임계/성능 모두 2019 test 기준. 학습은 train 2017–18.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
import csw_pipeline as P

OUT = HERE / "out" / "v3"; OUT.mkdir(parents=True, exist_ok=True)
C = HERE / "cache"
meta = json.loads((C / "meta_g.json").read_text())
df = pd.read_parquet(C / "features_g.parquet")
y = df["is_csw"].to_numpy(); yr = df["game_year"].to_numpy()
tr = np.isin(yr, P.TRAIN_YEARS); te = yr == 2019
trp, tep = np.where(tr)[0], np.where(te)[0]
rng = np.random.default_rng(0)
N = 45000
cap = lambda i, n=N: i if len(i) <= n else rng.choice(i, n, replace=False)
bp = json.loads((OUT / "E_optuna.json").read_text())["best_params"]

g_feats = [c for c in meta["g_feats"] if c in df.columns]
batter_all = set(meta["batter_all"])
no_batter = [c for c in g_feats if c not in batter_all]
basic_batter = no_batter + [c for c in (meta["groups"]["batter_scout"] + [meta.get("batter_te","batter_te")]) if c in df.columns]

def run(cols, target="is_csw", mask=None):
    X = P.build_matrix(df, cols)
    t = df[target].to_numpy()
    idx = trp if mask is None else np.intersect1d(trp, np.where(mask)[0])
    m = LGBMClassifier(**bp, random_state=0, n_jobs=-1, verbose=-1)
    f = cap(idx); m.fit(X.iloc[f], t[f])
    return m, X, m.predict_proba(X.iloc[tep])[:, 1]

res, t0 = {}, time.time()
for name, cols in [("1_no_batter", no_batter), ("2_batter_basic", basic_batter), ("3_batter_full", g_feats)]:
    _, _, p = run(cols)
    res[name] = {**P.metrics(y[tep], p), "n_feats": len(cols)}
    print(f"  {name:16s} logloss {res[name]['logloss']:.4f} auc {res[name]['roc_auc']:.4f} ({len(cols)} feats)")
    if name == "3_batter_full": p_single = p

# 4) 확장 타자 + CS/W 분해
sw = df["is_swing"].astype(bool).to_numpy()
mA, X, pA = run(g_feats, "is_swing")
mB, _, _ = run(g_feats, "is_whiff", sw)
mC, _, _ = run(g_feats, "is_called", ~sw)
Xte = X.iloc[tep]
pB = mB.predict_proba(Xte)[:, 1]; pC = mC.predict_proba(Xte)[:, 1]
pD = pA * pB + (1 - pA) * pC
res["4_batter_full_decomposed"] = P.metrics(y[tep], pD)
res["5_ensemble"] = P.metrics(y[tep], 0.5 * p_single + 0.5 * pD)
res["baselines"] = P.baselines(df, tr, te)
res["delta_batter_basic"] = round(res["2_batter_basic"]["logloss"] - res["1_no_batter"]["logloss"], 5)
res["delta_batter_full"] = round(res["3_batter_full"]["logloss"] - res["1_no_batter"]["logloss"], 5)

# 타자 피처 중요도(확장 모델)
mS = LGBMClassifier(**bp, random_state=0, n_jobs=-1, verbose=-1)
f = cap(trp); mS.fit(X.iloc[f], y[f])
imp = pd.Series(mS.feature_importances_, index=X.columns).sort_values(ascending=False)
bcols = [c for c in imp.index if c.startswith(("b_", "batter_"))]
res["batter_importance_share"] = round(float(imp[bcols].sum() / imp.sum()), 4)
res["top_batter_features"] = {k: int(v) for k, v in imp[bcols].head(12).items()}
res["top_overall"] = {k: int(v) for k, v in imp.head(12).items()}

(OUT / "G_batter.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
for k in ["4_batter_full_decomposed", "5_ensemble"]:
    print(f"  {k:26s} logloss {res[k]['logloss']:.4f} auc {res[k]['roc_auc']:.4f}")
print("타자 피처 중요도 점유율:", res["batter_importance_share"])
print(f"[G] {time.time()-t0:.1f}s")
