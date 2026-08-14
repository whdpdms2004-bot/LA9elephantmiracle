"""I) 모델 구조별 Optuna 튜닝 — 계열마다 독립 탐색 후 비교·앙상블.

계열: logreg / rf(랜덤포레스트) / et(엑스트라트리) / hgb / lgbm / xgb
· 피처: FULL(311) = 투수+타자+워크로드+PitchPredict
· 탐색: 2017 학습 → 2018 검증(LogLoss 최소화). SQLite 스터디로 이어달리기.
· 최종: 2017–18 재학습 → 2019 test 평가
실행: python run_i.py <family> [trials_target]      # 예: python run_i.py rf 12
      python run_i.py combine                      # 전 계열 비교 + 앙상블
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd, optuna
from sklearn.metrics import log_loss
import csw_pipeline as P

optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = HERE / "out" / "i"; OUT.mkdir(parents=True, exist_ok=True)
C = HERE / "cache"
meta = json.loads((C / "meta_g.json").read_text())
df = pd.read_parquet(C / "features_g.parquet")
y = df["is_csw"].to_numpy(); yr = df["game_year"].to_numpy()
tr = np.isin(yr, P.TRAIN_YEARS); te = yr == 2019
trp, tep = np.where(tr)[0], np.where(te)[0]
f17, v18 = np.where(yr == 2017)[0], np.where(yr == 2018)[0]
rng = np.random.default_rng(0)
FULL = [c for c in meta["h_feats"] if c in df.columns]
X = P.build_matrix(df, FULL)

TRIAL_FIT = {"logreg": 20000, "rf": 7000, "et": 7000, "hgb": 20000, "lgbm": 18000, "xgb": 18000}
FINAL_FIT = {"logreg": 45000, "rf": 20000, "et": 20000, "hgb": 45000, "lgbm": 45000, "xgb": 45000}


def build(fam, t=None, params=None):
    """t=Optuna trial(탐색) 또는 params=고정값(최종)."""
    sug = (lambda *a, **k: None)
    if t is not None:
        if fam == "logreg":
            params = dict(C=t.suggest_float("C", 1e-3, 10, log=True))
        elif fam in ("rf", "et"):
            params = dict(n_estimators=t.suggest_int("n_estimators", 80, 200, step=40),
                          max_depth=t.suggest_int("max_depth", 6, 20),
                          min_samples_leaf=t.suggest_int("min_samples_leaf", 20, 400, log=True),
                          max_features=t.suggest_float("max_features", 0.1, 0.8))
        elif fam == "hgb":
            params = dict(max_iter=t.suggest_int("max_iter", 150, 500, step=50),
                          learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                          max_leaf_nodes=t.suggest_int("max_leaf_nodes", 15, 127, log=True),
                          min_samples_leaf=t.suggest_int("min_samples_leaf", 50, 500, log=True),
                          l2_regularization=t.suggest_float("l2_regularization", 1e-3, 30, log=True))
        elif fam == "lgbm":
            params = dict(n_estimators=t.suggest_int("n_estimators", 150, 500, step=50),
                          learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                          num_leaves=t.suggest_int("num_leaves", 15, 127, log=True),
                          min_child_samples=t.suggest_int("min_child_samples", 50, 800, log=True),
                          subsample=t.suggest_float("subsample", 0.6, 1.0),
                          colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                          reg_lambda=t.suggest_float("reg_lambda", 1e-2, 30, log=True))
        else:  # xgb
            params = dict(n_estimators=t.suggest_int("n_estimators", 150, 500, step=50),
                          learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                          max_depth=t.suggest_int("max_depth", 3, 9),
                          min_child_weight=t.suggest_int("min_child_weight", 1, 30, log=True),
                          subsample=t.suggest_float("subsample", 0.6, 1.0),
                          colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                          reg_lambda=t.suggest_float("reg_lambda", 1e-2, 30, log=True))
    if fam == "logreg":
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=300, **params))]), params
    if fam in ("rf", "et"):
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        Cls = RandomForestClassifier if fam == "rf" else ExtraTreesClassifier
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", Cls(random_state=0, n_jobs=-1, **params))]), params
    if fam == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=0, **params), params
    if fam == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=0, n_jobs=-1, verbose=-1, **params), params
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=0, n_jobs=-1, tree_method="hist",
                         eval_metric="logloss", **params), params


def tune(fam, target_trials):
    db = f"sqlite:////tmp/optuna_i_{fam}.db"
    st = optuna.create_study(direction="minimize", study_name=fam, storage=db,
                             load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=0))
    fit = rng.choice(f17, min(TRIAL_FIT[fam], len(f17)), replace=False)
    def obj(t):
        m, _ = build(fam, t=t); m.fit(X.iloc[fit], y[fit])
        return log_loss(y[v18], m.predict_proba(X.iloc[v18])[:, 1], labels=[0, 1])
    done = len([x for x in st.trials if x.value is not None])
    if done < target_trials:
        st.optimize(obj, n_trials=target_trials - done, show_progress_bar=False)
    return st


def finalize(fam, st):
    m, params = build(fam, params=st.best_params)
    fit = rng.choice(trp, min(FINAL_FIT[fam], len(trp)), replace=False)
    m.fit(X.iloc[fit], y[fit])
    p = m.predict_proba(X.iloc[tep])[:, 1]
    np.save(OUT / f"p_{fam}.npy", p)
    res = {"family": fam, "best_params": st.best_params, "best_val_logloss": round(st.best_value, 5),
           "n_trials": len([x for x in st.trials if x.value is not None]),
           "history": [round(x.value, 5) for x in st.trials if x.value is not None],
           "test": P.metrics(y[tep], p), "n_feats": len(FULL)}
    (OUT / f"{fam}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    print(f"[{fam}] trials {res['n_trials']} | val {st.best_value:.5f} → TEST {res['test']['logloss']:.4f} auc {res['test']['roc_auc']:.4f}")


if __name__ == "__main__":
    arg = sys.argv[1]
    t0 = time.time()
    if arg == "combine":
        fams = [f.stem for f in OUT.glob("*.json") if f.stem in TRIAL_FIT]
        res = {f: json.loads((OUT / f"{f}.json").read_text()) for f in fams}
        tbl = {f: r["test"] for f, r in res.items()}
        # 상위 계열 확률 평균 앙상블
        ps = {f: np.load(OUT / f"p_{f}.npy") for f in fams}
        order = sorted(fams, key=lambda f: res[f]["test"]["logloss"])
        ens = {}
        for k in range(2, len(order) + 1):
            sel = order[:k]
            ens[f"top{k}_" + "+".join(sel)] = P.metrics(y[tep], np.mean([ps[f] for f in sel], axis=0))
        # FULL 분해 결과와도 결합
        fin = HERE / "out/final"
        if (fin / "p_decomp.npy").exists():
            pdz = np.load(fin / "p_decomp.npy")
            best_f = order[0]
            for w in (0.3, 0.5, 0.7):
                ens[f"{best_f}+decomp(w={w})"] = P.metrics(y[tep], w * ps[best_f] + (1 - w) * pdz)
        out = {"per_family": tbl, "params": {f: res[f]["best_params"] for f in fams},
               "trials": {f: res[f]["n_trials"] for f in fams},
               "val_logloss": {f: res[f]["best_val_logloss"] for f in fams},
               "ensembles": ens, "baselines": P.baselines(df, tr, te), "n_feats": len(FULL)}
        (OUT / "I_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
        print(pd.DataFrame(tbl).T[["logloss", "roc_auc", "pr_auc", "ece"]].sort_values("logloss").to_string())
        print("\n앙상블 best:", min(ens.items(), key=lambda kv: kv[1]["logloss"])[0],
              round(min(v["logloss"] for v in ens.values()), 4))
    else:
        target = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        st = tune(arg, target)
        done = len([x for x in st.trials if x.value is not None])
        if done >= target:
            finalize(arg, st)
        else:
            print(f"[{arg}] trials {done}/{target} best={st.best_value:.5f} (재실행하면 이어서)")
    print(f"{time.time()-t0:.1f}s")
