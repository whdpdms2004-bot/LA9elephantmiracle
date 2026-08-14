"""지금까지 만든 모델들의 F1 보고 (임계값은 2018에서 선택 → 2019에 적용).

주의: 기본 임계 0.5는 base rate 0.28인 문제에서 거의 아무것도 양성으로 안 잡아 F1이 붕괴한다.
따라서 (a) 0.5 고정, (b) 2018에서 F1 최대화한 임계, 둘 다 보고한다.
비교 기준: 전부 양성(all-positive) F1 = 2p/(1+p).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import csw_pipeline as P

OUT = HERE / "out" / "v3"
meta = json.loads((HERE / "cache/meta.json").read_text())
df = pd.read_parquet(HERE / "cache/features.parquet")
y = df["is_csw"].to_numpy()
yr = df["game_year"].to_numpy()
tr = np.isin(yr, P.TRAIN_YEARS); te = yr == 2019
val = yr == 2018                      # 임계 선택용 (train 내부)
fit17 = yr == 2017
trp, tep, valp, f17 = np.where(tr)[0], np.where(te)[0], np.where(val)[0], np.where(fit17)[0]
rng = np.random.default_rng(0)
cap = lambda idx, n: idx if len(idx) <= n else rng.choice(idx, n, replace=False)
bp = json.loads((OUT / "E_optuna.json").read_text())["best_params"]
X = P.build_matrix(df, meta["full_feats"])
Xd = P.build_matrix(df, meta["derived_feats"])


def best_thr(y_true, p):
    """F1 최대 임계 탐색."""
    ths = np.quantile(p, np.linspace(0.30, 0.95, 60))
    f = [f1_score(y_true, (p >= t).astype(int), zero_division=0) for t in ths]
    return float(ths[int(np.argmax(f))]), float(max(f))


def report(name, p_val, p_test, rows):
    thr, f1v = best_thr(y[valp], p_val)
    yt = y[tep]
    for tag, t in [("thr=0.5", 0.5), (f"thr*={thr:.3f}", thr)]:
        pred = (p_test >= t).astype(int)
        rows.append({"모델": name, "임계": tag,
                     "F1": round(f1_score(yt, pred, zero_division=0), 4),
                     "정밀도": round(precision_score(yt, pred, zero_division=0), 4),
                     "재현율": round(recall_score(yt, pred, zero_division=0), 4),
                     "정확도": round(accuracy_score(yt, pred), 4),
                     "양성예측률": round(float(pred.mean()), 4),
                     "AUC": round(roc_auc_score(yt, p_test), 4)})


def fit_pred(target, mask=None, model="lgbm", cols=None, n=28000):
    """2017로 학습→2018 예측(임계용) / train 전체로 학습→2019 예측(평가용)."""
    XX = X if cols is None else cols
    t = df[target].to_numpy()
    idx17 = f17 if mask is None else np.intersect1d(f17, np.where(mask)[0])
    idxtr = trp if mask is None else np.intersect1d(trp, np.where(mask)[0])
    def mk():
        return (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
                min_samples_leaf=200, l2_regularization=1.0, random_state=0) if model == "hgb"
                else LGBMClassifier(**bp, random_state=0, n_jobs=-1, verbose=-1))
    m1 = mk(); f = cap(idx17, n); m1.fit(XX.iloc[f], t[f])
    m2 = mk(); f = cap(idxtr, n); m2.fit(XX.iloc[f], t[f])
    return m1.predict_proba(XX.iloc[valp])[:, 1], m2.predict_proba(XX.iloc[tep])[:, 1], m2


rows = []
# 0) 기준선
p_all_v = np.ones(len(valp)); p_all_t = np.ones(len(tep))
yt = y[tep]
rows.append({"모델": "기준: 전부 양성", "임계": "-", "F1": round(f1_score(yt, np.ones_like(yt)), 4),
             "정밀도": round(float(yt.mean()), 4), "재현율": 1.0,
             "정확도": round(float(yt.mean()), 4), "양성예측률": 1.0, "AUC": 0.5})
# count-only
key = ["balls", "strikes", "stand", "p_throws"]
gm = df[tr].groupby(key)["is_csw"].mean()
pc_v = df.iloc[valp][key].merge(gm.rename("p"), on=key, how="left")["p"].fillna(y[trp].mean()).to_numpy()
pc_t = df.iloc[tep][key].merge(gm.rename("p"), on=key, how="left")["p"].fillna(y[trp].mean()).to_numpy()
report("기준: count-only", pc_v, pc_t, rows)

# 1) B라운드 최고 (HistGB, derived 205)
v, t_, _ = fit_pred("is_csw", model="hgb", cols=Xd, n=28000)
report("B: HistGB (205피처)", v, t_, rows)
# 2) E: full + optuna 단일
v, t_, _ = fit_pred("is_csw")
report("E: full+Optuna 단일", v, t_, rows)
p_single_v, p_single_t = v, t_
# 3) F: 분해
sw = df["is_swing"].astype(bool).to_numpy()
vA, tA, _ = fit_pred("is_swing")
vB, tB, mB = fit_pred("is_whiff", sw)
vC, tC, mC = fit_pred("is_called", ~sw)
# 결합: 전체 val/test에 대해 B,C 모델 예측 필요
vB_all = mB.predict_proba(X.iloc[valp])[:, 1]; tB_all = tB
vC_all = mC.predict_proba(X.iloc[valp])[:, 1]; tC_all = tC
pD_v = vA * vB_all + (1 - vA) * vC_all
pD_t = tA * tB_all + (1 - tA) * tC_all
report("F: CS/W 분해 결합", pD_v, pD_t, rows)
report("F: 앙상블(단일+분해)", 0.5*p_single_v + 0.5*pD_v, 0.5*p_single_t + 0.5*pD_t, rows)

out = pd.DataFrame(rows)
out.to_csv(OUT / "f1_report.csv", index=False, encoding="utf-8-sig")
print(out.to_string(index=False))
print("\nbase rate(2019):", round(float(yt.mean()), 4))
