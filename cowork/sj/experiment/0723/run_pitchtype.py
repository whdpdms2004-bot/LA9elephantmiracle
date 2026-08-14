"""C) 구종 예측 모델의 정보를 CSW 모델 입력으로 사용 (stacking).

아이디어: 예측 시점엔 현재 구종을 모른다(엄격 투구 전). 그래서 같은 투구 전 피처로
구종을 예측하는 보조 모델을 만들고, 그 **예측 확률분포(soft 구종 추정)** 를 CSW 모델 입력에 추가한다.
  · 추가 피처: pt_hat_<구종> (확률), pt_entropy(불확실성), pt_max(최댓값), pt_top1(예측 구종)
  · 피처 셀렉션: 구종 모델의 중요도 상위 K개를 CSW 모델 입력으로 함께 사용

누수 방지(핵심): CSW 모델의 train 행에는 **OOF(경기그룹 K-fold) 예측**을 붙인다.
전체 train으로 학습한 구종 모델을 train 행에 그대로 쓰면 자기 라벨을 본 값이 되어 누수.
test(2019) 행에는 train 전체로 학습한 구종 모델의 예측을 사용.

실행: python run_pitchtype.py build   # OOF 구종 확률 생성 → cache/pt_feats.parquet
      python run_pitchtype.py eval    # CSW 모델(기존 피처 +구종정보) 학습·평가 → out/cv/C_*.json
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import csw_pipeline as P

OUT = HERE / "out" / "v2"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = HERE / "cache"
PT_FILE = CACHE / "pt_feats.parquet"
TOP_CLASSES = ["FF", "SI", "SL", "CH", "CU", "FC"]     # 그 외는 OTHER
K = 3
CAP = 30_000
PT_TREES = 120


def _pt_label(s: pd.Series) -> pd.Series:
    return s.where(s.isin(TOP_CLASSES), "OTHER").fillna("OTHER")


def build():
    """구종 예측 모델 → OOF 확률(train) + full-train 모델 확률(test) 저장."""
    from lightgbm import LGBMClassifier
    meta = json.loads((CACHE / "meta.json").read_text())
    df = pd.read_parquet(CACHE / "features.parquet")
    ylab = _pt_label(df["pitch_type"].astype("string")).to_numpy()   # 캐시에 포함됨
    classes = sorted(set(ylab))

    feats = meta["derived_feats"]
    X = P.build_matrix(df, feats)
    tr = df["game_year"].isin(P.TRAIN_YEARS).to_numpy(); te = df["game_year"].eq(2019).to_numpy()
    tr_pos, te_pos = np.where(tr)[0], np.where(te)[0]
    rng = np.random.default_rng(0)

    proba = pd.DataFrame(0.0, index=df.index, columns=[f"pt_hat_{c}" for c in classes], dtype="float32")
    # --- train: OOF (경기그룹 K-fold) ---
    groups = df["game_pk"].to_numpy()[tr_pos]
    for ai, bi in GroupKFold(n_splits=K).split(tr_pos, ylab[tr_pos], groups):
        fit, val = tr_pos[ai], tr_pos[bi]
        if len(fit) > CAP: fit = rng.choice(fit, CAP, replace=False)
        m = LGBMClassifier(n_estimators=PT_TREES, learning_rate=0.08, num_leaves=63, min_child_samples=200,
                           subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1)
        m.fit(X.iloc[fit], ylab[fit])
        pr = m.predict_proba(X.iloc[val])
        for j, c in enumerate(m.classes_):
            proba.iloc[val, proba.columns.get_loc(f"pt_hat_{c}")] = pr[:, j].astype("float32")
    # --- test: full-train 모델 ---
    fit = tr_pos if len(tr_pos) <= CAP else rng.choice(tr_pos, CAP, replace=False)
    m = LGBMClassifier(n_estimators=PT_TREES, learning_rate=0.08, num_leaves=63, min_child_samples=200,
                       subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1)
    m.fit(X.iloc[fit], ylab[fit])
    pr = m.predict_proba(X.iloc[te_pos])
    for j, c in enumerate(m.classes_):
        proba.iloc[te_pos, proba.columns.get_loc(f"pt_hat_{c}")] = pr[:, j].astype("float32")

    pcols = list(proba.columns)
    pv = proba[pcols].to_numpy(dtype="float64")
    proba["pt_entropy"] = (-(pv * np.log(np.clip(pv, 1e-9, 1))).sum(axis=1)).astype("float32")
    proba["pt_max"] = pv.max(axis=1).astype("float32")
    # 구종 모델 정확도(참고) + 중요도 상위 피처
    acc = float((np.array([c.replace("pt_hat_", "") for c in pcols])[pv.argmax(axis=1)][te_pos] == ylab[te_pos]).mean())
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    proba.to_parquet(PT_FILE, index=False)
    (OUT / "C_pitchtype_model.json").write_text(json.dumps({
        "classes": classes, "test_top1_accuracy": round(acc, 4),
        "majority_baseline": round(float(pd.Series(ylab[te_pos]).value_counts(normalize=True).iloc[0]), 4),
        "n_pt_features": len(proba.columns),
        "top30_importance": {k: int(v) for k, v in imp.head(30).items()},
    }, ensure_ascii=False, indent=2))
    print(f"[build] saved {PT_FILE.name} | classes={classes} | test top1 acc={acc:.4f}")


def evaluate():
    """기존 피처 vs 기존+구종정보 → 경기그룹 K-fold CV + 2019 test 비교."""
    from lightgbm import LGBMClassifier
    meta = json.loads((CACHE / "meta.json").read_text())
    df = pd.read_parquet(CACHE / "features.parquet")
    pt = pd.read_parquet(PT_FILE)
    y = df["is_csw"].to_numpy()
    tr = df["game_year"].isin(P.TRAIN_YEARS).to_numpy(); te = df["game_year"].eq(2019).to_numpy()
    tr_pos, te_pos = np.where(tr)[0], np.where(te)[0]
    groups = df["game_pk"].to_numpy()[tr_pos]
    rng = np.random.default_rng(0)

    base = P.build_matrix(df, meta["derived_feats"])
    withpt = pd.concat([base, pt.set_index(base.index)], axis=1)
    res = {}
    for name, X in [("without_pitchtype", base), ("with_pitchtype", withpt)]:
        folds = []
        for ai, bi in GroupKFold(n_splits=K).split(tr_pos, y[tr_pos], groups):
            fit, val = tr_pos[ai], tr_pos[bi]
            if len(fit) > CAP: fit = rng.choice(fit, CAP, replace=False)
            m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                               subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, n_jobs=-1, verbose=-1)
            m.fit(X.iloc[fit], y[fit]); folds.append(P.metrics(y[val], m.predict_proba(X.iloc[val])[:, 1]))
        cvm = {k: round(float(np.mean([f[k] for f in folds])), 4) for k in ["logloss","brier","roc_auc","pr_auc","ece"]}
        fit = tr_pos if len(tr_pos) <= CAP else rng.choice(tr_pos, CAP, replace=False)
        m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, n_jobs=-1, verbose=-1)
        m.fit(X.iloc[fit], y[fit])
        res[name] = {"cv_mean": cvm, "test_2019": P.metrics(y[te], m.predict_proba(X.iloc[te_pos])[:, 1]),
                     "n_features": int(X.shape[1])}
        print(name, "CV", cvm["logloss"], "TEST", res[name]["test_2019"]["logloss"])
    res["delta_test_logloss"] = round(res["with_pitchtype"]["test_2019"]["logloss"]
                                      - res["without_pitchtype"]["test_2019"]["logloss"], 5)
    res["baselines"] = P.baselines(df, tr, te)
    (OUT / "C_compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    print("saved out/cv/C_compare.json | Δtest logloss =", res["delta_test_logloss"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    t = time.time(); (build if cmd == "build" else evaluate)(); print(f"{cmd} {time.time()-t:.1f}s")
