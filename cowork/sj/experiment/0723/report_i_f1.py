"""I 라운드 전 모델의 F1·정확도 보고.

임계값 선택: 2019(test)로 고르면 반칙이므로, 저장된 test 확률의 **분포 분위수**를 쓰지 않고
2018 검증 예측을 다시 만들어 고른다. 계열별 재학습(2017→2018) 필요.
기준선: 정확도는 '항상 아니다'(=1-base), F1은 '전부 CSW'(=2p/(1+p)).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             roc_auc_score, log_loss)
import csw_pipeline as P
import run_i as R           # build/tune 재사용

OUT = HERE / "out" / "i"
y, X, df = R.y, R.X, R.df
tep, v18, f17, trp = R.tep, R.v18, R.f17, R.trp
yt = y[tep]; yv = y[v18]
base = float(yt.mean())
rng = np.random.default_rng(0)
blk = sys.argv[1] if len(sys.argv) > 1 else "val"

if blk == "val":
    # 2018 검증 예측 생성 (계열별, 2017 학습) → 임계 선택용
    store = {}
    for fam in ["lgbm", "hgb", "xgb", "et", "rf", "logreg"]:
        fj = OUT / f"{fam}.json"
        if not fj.exists(): continue
        bp = json.loads(fj.read_text())["best_params"]
        m, _ = R.build(fam, params=bp)
        fit = rng.choice(f17, min(R.TRIAL_FIT[fam], len(f17)), replace=False)
        m.fit(X.iloc[fit], y[fit])
        store[fam] = m.predict_proba(X.iloc[v18])[:, 1]
        print(fam, "val done")
    np.savez(OUT / "val_probs.npz", **store)
    print("saved val_probs.npz")

else:
    V = dict(np.load(OUT / "val_probs.npz"))
    T = {f: np.load(OUT / f"p_{f}.npy") for f in V}
    # 분해 확률 (FULL 피처, 이미 저장됨) + 2018용 분해도 필요 → 근사: 계열 최고와 동일 가중
    fin = HERE / "out/final"
    pdz_t = np.load(fin / "p_decomp.npy") if (fin / "p_decomp.npy").exists() else None

    def pick_thr(pv, yv, metric):
        ths = np.quantile(pv, np.linspace(0.05, 0.95, 80))
        sc = [(f1_score(yv, (pv >= t).astype(int), zero_division=0) if metric == "f1"
               else accuracy_score(yv, (pv >= t).astype(int))) for t in ths]
        return float(ths[int(np.argmax(sc))])

    rows = []
    # 기준선
    rows.append({"모델": "기준: 전부 CSW", "F1": round(2*base/(1+base), 4), "정밀도": round(base, 4),
                 "재현율": 1.0, "정확도@F1임계": round(base, 4), "정확도@최적": round(base, 4),
                 "AUC": 0.5, "LogLoss": None})
    rows.append({"모델": "기준: 항상 아니다", "F1": 0.0, "정밀도": 0.0, "재현율": 0.0,
                 "정확도@F1임계": round(1-base, 4), "정확도@최적": round(1-base, 4),
                 "AUC": 0.5, "LogLoss": None})
    league = float(y[trp].mean())
    rows.append({"모델": "기준: 리그평균 확률", "F1": round(2*base/(1+base), 4), "정밀도": round(base,4),
                 "재현율": 1.0, "정확도@F1임계": round(base,4), "정확도@최적": round(1-base,4),
                 "AUC": 0.5, "LogLoss": round(log_loss(yt, np.full(len(yt), league), labels=[0,1]), 4)})

    def add(name, pv, pt):
        t_f1 = pick_thr(pv, yv, "f1"); t_ac = pick_thr(pv, yv, "acc")
        pf = (pt >= t_f1).astype(int)
        rows.append({"모델": name,
                     "F1": round(f1_score(yt, pf, zero_division=0), 4),
                     "정밀도": round(precision_score(yt, pf, zero_division=0), 4),
                     "재현율": round(recall_score(yt, pf, zero_division=0), 4),
                     "정확도@F1임계": round(accuracy_score(yt, pf), 4),
                     "정확도@최적": round(accuracy_score(yt, (pt >= t_ac).astype(int)), 4),
                     "AUC": round(roc_auc_score(yt, pt), 4),
                     "LogLoss": round(log_loss(yt, np.clip(pt, 1e-6, 1-1e-6), labels=[0,1]), 4)})

    order = sorted(V, key=lambda f: log_loss(yt, np.clip(T[f],1e-6,1-1e-6), labels=[0,1]))
    for f in order:
        add(f"I: {f}", V[f], T[f])
    # 계열 평균 앙상블
    for k in (2, 3):
        sel = order[:k]
        add(f"앙상블 top{k} ({'+'.join(sel)})",
            np.mean([V[f] for f in sel], axis=0), np.mean([T[f] for f in sel], axis=0))
    # 최고계열 + CS/W 분해 (test만 분해 확률 존재 → val은 최고계열로 임계 선택)
    if pdz_t is not None:
        bf = order[0]
        for w in (0.3, 0.5):
            add(f"최종: {bf}+분해(w={w})", V[bf], w*T[bf] + (1-w)*pdz_t)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "I_f1_accuracy.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"\n2019 base rate {base:.4f} | '항상 아니다' 정확도 {1-base:.4f} | '전부 CSW' F1 {2*base/(1+base):.4f}")
