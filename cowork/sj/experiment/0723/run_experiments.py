"""experiment 실행(블록별 체크포인트 → 45s 제한 대응, 재실행 시 이어서).
실행: python run_experiments.py <exp>   (exp: global_basic|global_derived|pitcher_basic|pitcher_derived)
완료되면 summary.json 생성."""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import csw_pipeline as P

EXP = sys.argv[1]
SUB_FIT, SUB_FINAL, OPTUNA_TRIALS, SHAP_N, MIN_TRAIN, ABL_SUB = 35000, 100000, 8, 1200, 4500, 50000
t0 = time.time()
cache = HERE / "cache"; meta = json.loads((cache / "meta.json").read_text())
df = pd.read_parquet(cache / "features.parquet")
rd = HERE / "out" / EXP; rd.mkdir(parents=True, exist_ok=True)
feat_cols = meta["basic_feats"] if EXP.endswith("basic") else meta["derived_feats"]
is_global = EXP.startswith("global")
y = df["is_csw"].to_numpy()
tr = df["game_year"].isin(P.TRAIN_YEARS).to_numpy(); te = df["game_year"].eq(2019).to_numpy()
val = df["game_year"].eq(2018).to_numpy(); fit = df["game_year"].eq(2017).to_numpy()
rng = np.random.default_rng(0)
cap = lambda m, n: (lambda idx: idx if len(idx) <= n else rng.choice(idx, n, replace=False))(np.where(m)[0])
def done(b): return (rd / f"{b}.json").exists()
def save(b, o): (rd / f"{b}.json").write_text(json.dumps(o, ensure_ascii=False, indent=2, default=float))
def load(b): return json.loads((rd / f"{b}.json").read_text())
from lightgbm import LGBMClassifier
def lgbm(**kw): return LGBMClassifier(**{**dict(n_estimators=300,learning_rate=0.05,num_leaves=63,
    min_child_samples=200,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,n_jobs=-1,verbose=-1), **kw})

if is_global:
    X = P.build_matrix(df, feat_cols); vi = np.where(val)[0]; fi = cap(fit, SUB_FIT)
    if not done("compare"):
        comp = {}
        for name, mdl in P.model_zoo().items():
            mdl.fit(X.iloc[fi], y[fi]); comp[name] = P.metrics(y[val], mdl.predict_proba(X.iloc[vi])[:,1])
        save("compare", comp); print("compare done", round(time.time()-t0,1))
    if not done("optuna"):
        st = P.optuna_tune_lgbm(X.values, y, fi, vi, n_trials=OPTUNA_TRIALS)
        save("optuna", {"best_params": st.best_params, "best_val_prauc": round(st.best_value,4),
                        "history": [round(t.value,4) for t in st.trials]}); print("optuna done", round(time.time()-t0,1))
    if not done("final") and done("optuna"):
        bp = load("optuna")["best_params"]; ffi = cap(tr, SUB_FINAL)
        m = lgbm(**bp); m.fit(X.iloc[ffi], y[ffi]); pt = m.predict_proba(X.iloc[np.where(te)[0]])[:,1]
        from sklearn.calibration import calibration_curve
        frac, mean = calibration_curve(y[te], pt, n_bins=10, strategy="quantile")
        save("final", {"test_metrics_tuned_lgbm": P.metrics(y[te], pt), "baselines": P.baselines(df, tr, te),
                       "calibration": {"pred":[round(float(x),4) for x in mean],"obs":[round(float(x),4) for x in frac]}})
        print("final done", round(time.time()-t0,1))
    if not done("shap") and done("optuna"):
        bp = load("optuna")["best_params"]; ffi = cap(tr, SUB_FINAL)
        m = lgbm(**bp); m.fit(X.iloc[ffi], y[ffi])
        fgmap = {c:g for g in ["situation","basic_history","ids","pitcher_hist","arsenal","release_rep"] for c in meta["groups"].get(g,[])}
        gof = P.map_columns_to_groups(X.columns, fgmap)
        samp = X.iloc[np.where(te)[0]].sample(min(SHAP_N, int(te.sum())), random_state=0)
        fimp, cimp = P.shap_by_category(m, samp, gof)
        fimp.head(30).to_csv(rd/"shap_top_features.csv"); cimp.to_csv(rd/"shap_by_category.csv")
        fig, ax = plt.subplots(1,2, figsize=(11,4))
        cimp.iloc[::-1].plot.barh(ax=ax[0], color="#4C72B0"); ax[0].set_title(f"{EXP}: SHAP by category")
        fimp.head(12).iloc[::-1].plot.barh(ax=ax[1], color="#55A868"); ax[1].set_title("Top-12 features")
        plt.tight_layout(); plt.savefig(rd/"shap.png", dpi=110); plt.close()
        save("shap", {"top10": {k:round(float(v),5) for k,v in fimp.head(10).items()},
                      "category": {k:round(float(v),5) for k,v in cimp.items()}}); print("shap done", round(time.time()-t0,1))
    if EXP.endswith("derived") and not done("ablation"):
        g = meta["groups"]
        stages = [("count_only",g["count_only"]),("+situation",g["situation"]),("+basic_hist",g["basic_history"]),
                  ("+ids",g["ids"]),("+pitcher_hist",g["pitcher_hist"]),("+arsenal",g["arsenal"]),("+release_rep(full)",g["release_rep"])]
        ab = P.staged_ablation(df, stages, cap(tr, ABL_SUB), np.where(te)[0]); ab.to_csv(rd/"ablation.csv")
        save("ablation", ab.to_dict("index")); print("ablation done", round(time.time()-t0,1))
    blocks = ["compare","optuna","final","shap"] + (["ablation"] if EXP.endswith("derived") else [])
else:
    if not done("perpitcher"):
        resp, pred, src = P.per_pitcher_eval(df, feat_cols, tr, te, min_train=MIN_TRAIN, global_fit_cap=SUB_FINAL)
        save("perpitcher", resp); print("perpitcher done", round(time.time()-t0,1))
    if not done("globalref"):
        X = P.build_matrix(df, feat_cols); ffi = cap(tr, SUB_FINAL)
        m = lgbm(n_estimators=400); m.fit(X.iloc[ffi], y[ffi])
        save("globalref", {"global_reference_test": P.metrics(y[te], m.predict_proba(X.iloc[np.where(te)[0]])[:,1]),
                           "baselines": P.baselines(df, tr, te)}); print("globalref done", round(time.time()-t0,1))
    blocks = ["perpitcher","globalref"]

# 모든 블록 완료 시 summary 조립
if all(done(b) for b in blocks):
    summ = {"exp": EXP, "n_features_raw": len(feat_cols),
            "eval_protocol": "prequential/online (history features updated with 2019 past; model params frozen on 2017-18)",
            "min_train_perpitcher": MIN_TRAIN}
    for b in blocks: summ[b] = load(b)
    save("summary", summ)
    print(f"[{EXP}] ALL DONE {round(time.time()-t0,1)}s")
else:
    print(f"[{EXP}] partial: done={[b for b in blocks if done(b)]} remaining={[b for b in blocks if not done(b)]}")
