"""최종(FULL) 모델: 투수+타자+워크로드+PitchPredict 피처 (311) + Optuna 파라미터 + CS/W 분해.
블록 체크포인트(45s 제한 대응). 실행: python run_final.py <block>
block ∈ {single, decomp, combine}
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
import csw_pipeline as P

OUT = HERE / "out" / "final"; OUT.mkdir(parents=True, exist_ok=True)
C = HERE / "cache"
meta = json.loads((C / "meta_g.json").read_text())
df = pd.read_parquet(C / "features_g.parquet")
y = df["is_csw"].to_numpy(); yr = df["game_year"].to_numpy()
tr = np.isin(yr, P.TRAIN_YEARS); te = yr == 2019
trp, tep = np.where(tr)[0], np.where(te)[0]
rng = np.random.default_rng(0)
N = 45000                                  # G와 동일 조건
bp = json.loads((HERE / "out/v3/E_optuna.json").read_text())["best_params"]

FULL = [c for c in meta["h_feats"] if c in df.columns]     # 투수+타자+워크로드+PP
X = P.build_matrix(df, FULL)


def fit(target, mask=None):
    t = df[target].to_numpy()
    idx = trp if mask is None else np.intersect1d(trp, np.where(mask)[0])
    f = idx if len(idx) <= N else rng.choice(idx, N, replace=False)
    m = LGBMClassifier(**bp, random_state=0, n_jobs=-1, verbose=-1)
    m.fit(X.iloc[f], t[f]); return m


def save(name, obj): (OUT / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float))
def load(name): return json.loads((OUT / f"{name}.json").read_text())

blk = sys.argv[1] if len(sys.argv) > 1 else "single"
t0 = time.time()

if blk == "single":
    m = fit("is_csw"); p = m.predict_proba(X.iloc[tep])[:, 1]
    np.save(OUT / "p_single.npy", p)
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    grp = {}
    for g in ["situation","basic_history","ids","pitcher_hist","arsenal","release_rep",
              "batter_scout","batter_hist","gameflow","prior_ab","lineup","workload"]:
        cols = set(meta["groups"].get(g, []))
        sel = [c for c in X.columns if c in cols or any(c.startswith(o+"_") for o in cols)]
        grp[g] = int(imp[sel].sum()) if sel else 0
    save("single", {"n_feats": len(FULL), "test": P.metrics(y[tep], p),
                    "importance_by_group": grp, "top20": {k: int(v) for k, v in imp.head(20).items()}})
    print("single:", P.metrics(y[tep], p))

elif blk == "decomp":
    sw = df["is_swing"].astype(bool).to_numpy()
    Xte = X.iloc[tep]
    pA = fit("is_swing").predict_proba(Xte)[:, 1]
    pB = fit("is_whiff", sw).predict_proba(Xte)[:, 1]
    pC = fit("is_called", ~sw).predict_proba(Xte)[:, 1]
    pD = pA * pB + (1 - pA) * pC
    np.save(OUT / "p_decomp.npy", pD)
    save("decomp", {"test": P.metrics(y[tep], pD),
                    "mean_p_swing": float(pA.mean()), "mean_p_whiff": float(pB.mean()),
                    "mean_p_called": float(pC.mean())})
    print("decomp:", P.metrics(y[tep], pD))

else:  # combine
    ps = np.load(OUT / "p_single.npy"); pd_ = np.load(OUT / "p_decomp.npy")
    res = {"FULL_single": load("single")["test"], "FULL_decomposed": load("decomp")["test"]}
    best_w, best = None, 9
    for w in np.arange(0, 1.01, 0.1):
        m = P.metrics(y[tep], w * ps + (1 - w) * pd_)
        if m["logloss"] < best: best, best_w = m["logloss"], float(w)
    res["FULL_ensemble_best"] = P.metrics(y[tep], best_w * ps + (1 - best_w) * pd_)
    res["ensemble_weight_single"] = best_w
    res["baselines"] = P.baselines(df, tr, te)
    res["importance_by_group"] = load("single")["importance_by_group"]
    res["top20"] = load("single")["top20"]
    res["n_feats"] = load("single")["n_feats"]
    # 이전 라운드 대비
    res["history"] = {"A/B_205feat": 0.5760, "E_full_optuna": 0.5734, "F_decomp": 0.5722,
                      "G_batter_decomp": 0.5720,
                      "FINAL_all_decomp": res["FULL_decomposed"]["logloss"],
                      "FINAL_ensemble": res["FULL_ensemble_best"]["logloss"]}
    save("FINAL", res)
    for k in ["FULL_single", "FULL_decomposed", "FULL_ensemble_best"]:
        print(f"{k:22s} logloss {res[k]['logloss']:.4f} auc {res[k]['roc_auc']:.4f} ece {res[k]['ece']:.4f}")
    print("ensemble weight(single):", best_w)
print(f"[{blk}] {time.time()-t0:.1f}s")
