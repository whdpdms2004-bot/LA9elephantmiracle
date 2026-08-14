"""경기그룹 번갈아 K-fold 모델선택 + 2019 최종평가.

설계(확정):
  · 예측 시점: 엄격 투구 전 (현재 투구 물리·위치·구종·결과 전부 금지)
  · 분할: train 2017–18 안에서 GroupKFold(group=game_pk) K폴드 = 경기 단위로 묶어 번갈아
          → 경기 내 인접투구 누수 없음. 모델 '선택'에만 사용.
  · 최종: 선택한 모델을 train 100%로 재학습 → test 2019 평가 (test는 고정)
  · 구성: global(투수=ID/이력 피처) vs per_pitcher(투수별 개별 모델 + 임계미달 폴백)
  · 피처: 사용 가능한 전부(derived 135)
  · 모델: logreg / hgb / lgbm / xgb

주의(정직): 번갈아 K-fold는 시간을 뒤섞으므로, fit 행의 이력피처(_2w/_szn)가
val 경기 결과를 포함할 수 있다(내삽). 따라서 CV 점수는 2019 test보다 낙관적일 수 있으며,
최종 판단은 항상 test 2019로 한다.

실행: python run_cv.py <config>_<model>   예) python run_cv.py global_lgbm
      python run_cv.py all               (블록 순차, 45s 제한 시 재실행하면 이어서)
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import csw_pipeline as P

K_FOLDS      = 5
CAP_FOLD_FIT = 40_000     # 폴드당 학습 상한(속도/메모리)
CAP_FINAL    = 70_000     # 최종 전체학습 상한
MIN_TRAIN_PP = 4_500      # 투수별 모델 임계
PP_FOLDS     = 2          # 투수별 CV 폴드(계산량 제약)
PP_TREES     = 120        # 투수별 모델 트리 수(소표본이라 작게)
MODELS       = ["logreg", "hgb", "lgbm", "xgb"]
CONFIGS      = ["global", "per_pitcher"]
OUT = HERE / "out" / "v2"; OUT.mkdir(parents=True, exist_ok=True)


def get_model(name, seed=0, light=False):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    n = PP_TREES if light else 300
    leaves = 31 if light else 63
    if name == "logreg":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=400, C=1.0))])
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=n, learning_rate=0.06, max_leaf_nodes=leaves,
                min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    if name == "lgbm":
        return LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=leaves,
                min_child_samples=100 if light else 200, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
    if name == "xgb":
        return XGBClassifier(n_estimators=n, learning_rate=0.05, max_depth=4 if light else 5,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=seed, n_jobs=-1, tree_method="hist", eval_metric="logloss")
    raise ValueError(name)


def load():
    meta = json.loads((HERE / "cache/meta.json").read_text())
    df = pd.read_parquet(HERE / "cache/features.parquet")
    feats = meta["derived_feats"]                      # 사용 가능한 전부
    X = P.build_matrix(df, feats)
    y = df["is_csw"].to_numpy()
    tr = df["game_year"].isin(P.TRAIN_YEARS).to_numpy()
    te = df["game_year"].eq(2019).to_numpy()
    return meta, df, X, y, tr, te


def run_global(model_name, seed=0):
    meta, df, X, y, tr, te = load()
    tr_pos, te_pos = np.where(tr)[0], np.where(te)[0]
    groups = df["game_pk"].to_numpy()[tr_pos]
    rng = np.random.default_rng(seed)
    gkf = GroupKFold(n_splits=K_FOLDS)
    fold_rows = []
    for k, (ai, bi) in enumerate(gkf.split(tr_pos, y[tr_pos], groups), 1):
        fit = tr_pos[ai]; val = tr_pos[bi]
        if len(fit) > CAP_FOLD_FIT: fit = rng.choice(fit, CAP_FOLD_FIT, replace=False)
        m = get_model(model_name, seed)
        m.fit(X.iloc[fit], y[fit])
        p = m.predict_proba(X.iloc[val])[:, 1]
        fold_rows.append({**P.metrics(y[val], p), "fold": k, "n_fit": int(len(fit))})
    cv = pd.DataFrame(fold_rows)
    # 최종: train 100%(상한) 재학습 → test 2019
    fit = tr_pos if len(tr_pos) <= CAP_FINAL else rng.choice(tr_pos, CAP_FINAL, replace=False)
    m = get_model(model_name, seed); m.fit(X.iloc[fit], y[fit])
    test = P.metrics(y[te], m.predict_proba(X.iloc[te_pos])[:, 1])
    return {"config": "global", "model": model_name, "k_folds": K_FOLDS,
            "cv_folds": fold_rows,
            "cv_mean": {c: round(float(cv[c].mean()), 4) for c in ["logloss","brier","roc_auc","pr_auc","ece"]},
            "cv_std_logloss": round(float(cv["logloss"].std()), 4),
            "test_2019": test, "n_features": int(X.shape[1])}


def run_per_pitcher(model_name, seed=0):
    meta, df, X, y, tr, te = load()
    tri, tei = df.index[tr], df.index[te]
    counts = df.loc[tri].groupby("pitcher").size()
    elig = counts[counts >= MIN_TRAIN_PP].index
    rng = np.random.default_rng(seed)

    # --- CV: 투수별로 자기 경기들을 그룹 K-fold (모델 선택용) ---
    cv_p, cv_y = [], []
    for pid in elig:
        idx = tri[df.loc[tri, "pitcher"].eq(pid).to_numpy()]
        g = df.loc[idx, "game_pk"].to_numpy()
        if len(np.unique(g)) < PP_FOLDS or df.loc[idx, "is_csw"].nunique() < 2: continue
        gkf = GroupKFold(n_splits=PP_FOLDS)
        pos = df.index.get_indexer(idx)
        for ai, bi in gkf.split(pos, y[pos], g):
            f, v = pos[ai], pos[bi]
            if y[f].min() == y[f].max(): continue
            m = get_model(model_name, seed, light=True)
            m.fit(X.iloc[f], y[f])
            cv_p.append(m.predict_proba(X.iloc[v])[:, 1]); cv_y.append(y[v])
    cv_mean = P.metrics(np.concatenate(cv_y), np.concatenate(cv_p))

    # --- 최종: 각 투수 train 100%로 재학습 → 2019 (임계미달은 global 폴백) ---
    tr_pos, te_pos = np.where(tr)[0], np.where(te)[0]
    gfit = tr_pos if len(tr_pos) <= CAP_FINAL else rng.choice(tr_pos, CAP_FINAL, replace=False)
    gm = get_model(model_name, seed); gm.fit(X.iloc[gfit], y[gfit])
    pred = pd.Series(gm.predict_proba(X.iloc[te_pos])[:, 1], index=tei)
    src = pd.Series("global", index=tei)
    for pid in elig:
        ptr = tri[df.loc[tri, "pitcher"].eq(pid).to_numpy()]
        pte = tei[df.loc[tei, "pitcher"].eq(pid).to_numpy()]
        if len(pte) == 0 or df.loc[ptr, "is_csw"].nunique() < 2: continue
        m = get_model(model_name, seed, light=True)
        m.fit(X.loc[ptr], df.loc[ptr, "is_csw"].to_numpy())
        pred.loc[pte] = m.predict_proba(X.loc[pte])[:, 1]; src.loc[pte] = "per_pitcher"
    cov = float((src == "per_pitcher").mean())
    test = P.metrics(df.loc[tei, "is_csw"].to_numpy(), pred.to_numpy())
    return {"config": "per_pitcher", "model": model_name, "k_folds": PP_FOLDS,
            "cv_mean": cv_mean, "test_2019": test,
            "coverage_per_pitcher": round(cov, 4), "fallback_global": round(1 - cov, 4),
            "n_eligible": int(len(elig)), "min_train": MIN_TRAIN_PP, "n_features": int(X.shape[1])}


def block(name):
    cfg, mdl = name.rsplit("_", 1)
    f = OUT / f"{name}.json"
    if f.exists(): return False
    t = time.time()
    res = run_global(mdl) if cfg == "global" else run_per_pitcher(mdl)
    res["seconds"] = round(time.time() - t, 1)
    f.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    print(f"[{name}] done {res['seconds']}s | CV logloss {res['cv_mean']['logloss']} | TEST {res['test_2019']['logloss']}")
    return True


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = [f"{c}_{m}" for c in CONFIGS for m in MODELS] if arg == "all" else [arg]
    todo = [n for n in names if not (OUT / f"{n}.json").exists()]
    if not todo:
        print("모든 블록 완료:", [n for n in names]); sys.exit(0)
    for n in todo:
        block(n)
    rem = [n for n in names if not (OUT / f"{n}.json").exists()]
    print("남은 블록:", rem if rem else "없음")
