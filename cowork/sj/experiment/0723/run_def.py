"""D(EDA) / E(피처확장+튜닝) / F(CS·W 분해) 실험. 블록 체크포인트.
실행: python run_def.py <block>   block ∈ {eda, E_ablation, E_optuna, F_decomp}
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import csw_pipeline as P

OUT = HERE / "out" / "v3"; OUT.mkdir(parents=True, exist_ok=True)
CAP, KF = 45_000, 4
meta = json.loads((HERE / "cache/meta.json").read_text())
df = pd.read_parquet(HERE / "cache/features.parquet")
y = df["is_csw"].to_numpy()
tr = df["game_year"].isin(P.TRAIN_YEARS).to_numpy(); te = df["game_year"].eq(2019).to_numpy()
tr_pos, te_pos = np.where(tr)[0], np.where(te)[0]
rng = np.random.default_rng(0)
cap = lambda idx, n=CAP: idx if len(idx) <= n else rng.choice(idx, n, replace=False)


def lgbm(**kw):
    from lightgbm import LGBMClassifier
    d = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, n_jobs=-1, verbose=-1)
    d.update(kw); return LGBMClassifier(**d)


def fit_eval(X, target=None, mask=None, params=None):
    """train→test 평가. mask가 주어지면 해당 부분집합만 학습/평가."""
    t = df[target].to_numpy() if target else y
    trp = tr_pos if mask is None else np.intersect1d(tr_pos, np.where(mask)[0])
    tep = te_pos if mask is None else np.intersect1d(te_pos, np.where(mask)[0])
    m = lgbm(**(params or {})); f = cap(trp)
    m.fit(X.iloc[f], t[f])
    p = m.predict_proba(X.iloc[tep])[:, 1]
    return m, p, tep, P.metrics(t[tep], p)


# ─────────────────────────── D: EDA ───────────────────────────
def eda():
    g = meta["groups"]
    out = {}
    d = df.copy()
    d["count"] = d["balls"].astype(int).astype(str) + "-" + d["strikes"].astype(int).astype(str)
    out["by_count"] = d.groupby("count")["is_csw"].agg(["mean", "size"]).sort_values("mean").to_dict("index")
    out["by_strikes"] = d.groupby("strikes")["is_csw"].mean().to_dict()
    out["by_inning"] = d[d["inning"] <= 9].groupby("inning")["is_csw"].mean().to_dict()
    out["by_matchup"] = d.groupby("matchup")["is_csw"].mean().to_dict()
    pc = d.groupby("pitcher")["is_csw"].agg(["mean", "size"])
    out["pitcher_spread"] = {"min": float(pc["mean"].min()), "max": float(pc["mean"].max()),
                             "sd": float(pc["mean"].std()), "n": int(len(pc))}
    # 구성요소 분해
    out["components"] = {"csw": float(d.is_csw.mean()), "called": float(d.is_called.mean()),
                         "whiff": float(d.is_whiff.mean()), "swing": float(d.is_swing.mean()),
                         "whiff_given_swing": float(d.loc[d.is_swing.eq(1), "is_whiff"].mean()),
                         "called_given_take": float(d.loc[d.is_take.eq(1), "is_called"].mean())}
    # 상관: 각 피처와 타깃
    num = [c for c in meta["full_feats"] if pd.api.types.is_numeric_dtype(df[c])]
    corr = df[num].corrwith(df["is_csw"]).abs().sort_values(ascending=False)
    out["top_corr"] = {k: round(float(v), 4) for k, v in corr.head(25).items()}
    (OUT / "eda.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
    print("[eda] saved. CSW by count(top3):", list(out["by_count"].items())[:3])


# ─────────────────────────── E: 피처 확장 ablation ───────────────────────────
def E_ablation():
    g = meta["groups"]
    stages = [
        ("1_count_only", g["count_only"]),
        ("2_situation", g["situation"]),
        ("3_basic_hist", g["basic_history"] + g["ids"]),
        ("4_pitcher_hist", g["pitcher_hist"]),
        ("5_arsenal_release", g["arsenal"] + g["release_rep"]),
        ("6_batter_scout(PP)", g["batter_scout"]),
        ("7_gameflow(PP)", g["gameflow"]),
        ("8_prior_ab+lineup(PP)", g["prior_ab"] + g["lineup"]),
    ]
    cum, rows = [], {}
    for name, cols in stages:
        cum = sorted(set(cum) | set(c for c in cols if c in df.columns))
        X = P.build_matrix(df, cum)
        _, _, _, mt = fit_eval(X)
        rows[name] = {**mt, "n_feats": int(X.shape[1])}
        print(f"  {name:24s} test logloss {mt['logloss']:.4f} auc {mt['roc_auc']:.4f} ({X.shape[1]} feats)")
    (OUT / "E_ablation.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=float))


# ─────────────────────────── E: Optuna 강화 ───────────────────────────
def E_optuna(n_trials=30):
    import optuna
    from lightgbm import LGBMClassifier
    from sklearn.metrics import log_loss
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X = P.build_matrix(df, meta["full_feats"])
    groups = df["game_pk"].to_numpy()[tr_pos]
    folds = list(GroupKFold(n_splits=3).split(tr_pos, y[tr_pos], groups))
    def obj(t):
        pr = dict(n_estimators=t.suggest_int("n_estimators", 200, 700, step=100),
                  learning_rate=t.suggest_float("learning_rate", 0.02, 0.12, log=True),
                  num_leaves=t.suggest_int("num_leaves", 31, 255, log=True),
                  min_child_samples=t.suggest_int("min_child_samples", 50, 800, log=True),
                  subsample=t.suggest_float("subsample", 0.6, 1.0),
                  colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                  reg_lambda=t.suggest_float("reg_lambda", 1e-2, 30, log=True),
                  reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True))
        sc = []
        for ai, bi in folds[:2]:
            f, v = cap(tr_pos[ai], 30000), tr_pos[bi]
            m = LGBMClassifier(**pr, random_state=0, n_jobs=-1, verbose=-1)
            m.fit(X.iloc[f], y[f]); sc.append(log_loss(y[v], m.predict_proba(X.iloc[v])[:, 1], labels=[0,1]))
        return float(np.mean(sc))
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    _, _, _, mt = fit_eval(X, params=st.best_params)
    res = {"best_params": st.best_params, "best_cv_logloss": round(st.best_value, 5),
           "n_trials": n_trials, "test_2019": mt,
           "history": [round(t.value, 5) for t in st.trials if t.value is not None],
           "baselines": P.baselines(df, tr, te)}
    (OUT / "E_optuna.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    print(f"[E_optuna] best CV {st.best_value:.5f} → TEST {mt['logloss']:.4f} (auc {mt['roc_auc']:.4f})")


# ─────────────────────────── F: CS / W 분해 ───────────────────────────
def F_decomp():
    X = P.build_matrix(df, meta["full_feats"])
    swing = df["is_swing"].astype(bool).to_numpy()
    res = {}
    # 단일 모델(기준)
    _, p_single, tep, m_single = fit_eval(X)
    res["single_is_csw"] = m_single
    # A: P(swing)
    mA, pA, _, mtA = fit_eval(X, target="is_swing")
    res["A_swing"] = mtA
    # B: P(whiff | swing)
    mB, pB, tepB, mtB = fit_eval(X, target="is_whiff", mask=swing)
    res["B_whiff_given_swing"] = mtB
    # C: P(called | take)
    mC, pC, tepC, mtC = fit_eval(X, target="is_called", mask=~swing)
    res["C_called_given_take"] = mtC
    # 결합: 전체 test에 대해 세 모델 예측
    Xte = X.iloc[te_pos]
    ps = mA.predict_proba(Xte)[:, 1]
    pw = mB.predict_proba(Xte)[:, 1]
    pc = mC.predict_proba(Xte)[:, 1]
    p_comb = ps * pw + (1 - ps) * pc
    res["combined_decomposed"] = P.metrics(y[te_pos], p_comb)
    # 앙상블(단일 + 분해 평균)
    res["ensemble_single_plus_decomp"] = P.metrics(y[te_pos], 0.5 * p_single + 0.5 * p_comb)
    res["baselines"] = P.baselines(df, tr, te)
    (OUT / "F_decomp.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    for k in ["single_is_csw", "combined_decomposed", "ensemble_single_plus_decomp"]:
        print(f"  {k:30s} logloss {res[k]['logloss']:.4f} auc {res[k]['roc_auc']:.4f}")


if __name__ == "__main__":
    b = sys.argv[1]
    f = (OUT / f"{ {'eda':'eda','E_ablation':'E_ablation','E_optuna':'E_optuna','F_decomp':'F_decomp'}[b] }.json")
    if f.exists(): print(f"{b} 이미 완료"); sys.exit(0)
    t = time.time()
    {"eda": eda, "E_ablation": E_ablation, "E_optuna": E_optuna, "F_decomp": F_decomp}[b]()
    print(f"[{b}] {time.time()-t:.1f}s")
