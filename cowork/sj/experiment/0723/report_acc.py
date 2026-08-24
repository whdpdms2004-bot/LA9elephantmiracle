"""정확도(정답 확률) 관점 보고 + 확률 보정(calibration) 점검.

핵심 비교 기준
  · 다수클래스(항상 '아니다') 정확도 = 1 - base rate = 0.717  ← 이걸 넘는지가 관건
  · 임계는 2018에서 '정확도 최대'로 선택 → 2019 적용
추가: 현재 투구 위치를 넣은 모델(투구 후 정보)과 대조 → 격차 정량화
"""
from __future__ import annotations
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from lightgbm import LGBMClassifier
import csw_pipeline as P

OUT = HERE / "out" / "v3"
meta = json.loads((HERE / "cache/meta.json").read_text())
df = pd.read_parquet(HERE / "cache/features.parquet")
y = df["is_csw"].to_numpy(); yr = df["game_year"].to_numpy()
tr = np.isin(yr, P.TRAIN_YEARS); te = yr == 2019
trp, tep, valp, f17 = np.where(tr)[0], np.where(te)[0], np.where(yr == 2018)[0], np.where(yr == 2017)[0]
rng = np.random.default_rng(0)
cap = lambda i, n: i if len(i) <= n else rng.choice(i, n, replace=False)
bp = json.loads((OUT / "E_optuna.json").read_text())["best_params"]
X = P.build_matrix(df, meta["full_feats"])
yt = y[tep]
N = 28000

def best_acc_thr(yv, pv):
    ths = np.quantile(pv, np.linspace(0.05, 0.95, 60))
    a = [accuracy_score(yv, (pv >= t).astype(int)) for t in ths]
    return float(ths[int(np.argmax(a))])

rows = []
def add(name, pv, pt):
    thr = best_acc_thr(y[valp], pv)
    rows.append({"모델": name,
                 "정확도@0.5": round(accuracy_score(yt, (pt >= 0.5).astype(int)), 4),
                 "정확도@최적": round(accuracy_score(yt, (pt >= thr).astype(int)), 4),
                 "최적임계": round(thr, 3),
                 "AUC": round(roc_auc_score(yt, pt), 4),
                 "LogLoss": round(log_loss(yt, np.clip(pt, 1e-6, 1-1e-6), labels=[0,1]), 4),
                 "Brier": round(brier_score_loss(yt, pt), 4)})

# 0) 기준선
base = float(yt.mean())
rows.append({"모델": "기준: 항상 '아니다'", "정확도@0.5": round(1-base, 4), "정확도@최적": round(1-base, 4),
             "최적임계": None, "AUC": 0.5, "LogLoss": None, "Brier": round(base*(1-base)+0, 4)})
rows.append({"모델": "기준: 리그평균 확률(0.283)", "정확도@0.5": round(1-base, 4), "정확도@최적": round(1-base, 4),
             "최적임계": None, "AUC": 0.5,
             "LogLoss": round(log_loss(yt, np.full(len(yt), y[trp].mean()), labels=[0,1]), 4),
             "Brier": round(brier_score_loss(yt, np.full(len(yt), y[trp].mean())), 4)})

def fit_pred(target, mask=None, XX=None):
    XX = X if XX is None else XX
    t = df[target].to_numpy()
    i17 = f17 if mask is None else np.intersect1d(f17, np.where(mask)[0])
    itr = trp if mask is None else np.intersect1d(trp, np.where(mask)[0])
    m1 = LGBMClassifier(**bp, random_state=0, n_jobs=-1, verbose=-1); f = cap(i17, N); m1.fit(XX.iloc[f], t[f])
    m2 = LGBMClassifier(**bp, random_state=0, n_jobs=-1, verbose=-1); f = cap(itr, N); m2.fit(XX.iloc[f], t[f])
    return m1.predict_proba(XX.iloc[valp])[:,1], m2.predict_proba(XX.iloc[tep])[:,1], m1, m2

# 1) 최고 투구 전 모델 = CS/W 분해
sw = df["is_swing"].astype(bool).to_numpy()
vA, tA, _, _ = fit_pred("is_swing")
vB, tB, mB1, mB2 = fit_pred("is_whiff", sw)
vC, tC, mC1, mC2 = fit_pred("is_called", ~sw)
pv = vA*mB1.predict_proba(X.iloc[valp])[:,1] + (1-vA)*mC1.predict_proba(X.iloc[valp])[:,1]
pt = tA*mB2.predict_proba(X.iloc[tep])[:,1] + (1-tA)*mC2.predict_proba(X.iloc[tep])[:,1]
add("투구 전 최고 (CS/W 분해)", pv, pt)
cal_p, cal_y = pt, yt

# 2) 대조: 현재 투구 위치 포함 (투구 후 정보)
COLS = ["game_year","game_type","pitcher","game_date","game_pk","at_bat_number","pitch_number",
        "description","type","zone","balls","strikes","stand","p_throws","plate_x","plate_z","sz_top","sz_bot"]
keep = set(df["pitcher"].unique().tolist())
parts = []
for b in pq.ParquetFile(P.RAW).iter_batches(batch_size=250_000, columns=COLS):
    d = b.to_pandas(); d = d[d["game_type"].eq("R") & d["pitcher"].isin(keep)]
    if len(d): parts.append(d)
raw = pd.concat(parts, ignore_index=True).sort_values(P.SEQ, kind="stable").reset_index(drop=True)
raw = P.add_labels(raw)
yr2 = raw["game_year"].to_numpy()
L = raw[["balls","strikes","plate_x","plate_z","sz_top","sz_bot"]].apply(pd.to_numeric, errors="coerce").astype("float32")
for c in ["stand","p_throws"]: L[c] = raw[c].astype("category")
y2 = raw["is_csw"].to_numpy()
i17b, itrb, ivb, iteb = np.where(yr2==2017)[0], np.where(np.isin(yr2,(2017,2018)))[0], np.where(yr2==2018)[0], np.where(yr2==2019)[0]
m1 = LGBMClassifier(**bp, random_state=0, n_jobs=-1, verbose=-1); f = cap(i17b, N); m1.fit(L.iloc[f], y2[f])
m2 = LGBMClassifier(**bp, random_state=0, n_jobs=-1, verbose=-1); f = cap(itrb, N); m2.fit(L.iloc[f], y2[f])
pv2, pt2 = m1.predict_proba(L.iloc[ivb])[:,1], m2.predict_proba(L.iloc[iteb])[:,1]
thr2 = None
ths = np.quantile(pv2, np.linspace(0.05,0.95,60)); a=[accuracy_score(y2[ivb],(pv2>=t).astype(int)) for t in ths]
thr2 = float(ths[int(np.argmax(a))])
rows.append({"모델": "대조: +현재 투구 위치 (투구 후)",
             "정확도@0.5": round(accuracy_score(y2[iteb], (pt2>=0.5).astype(int)), 4),
             "정확도@최적": round(accuracy_score(y2[iteb], (pt2>=thr2).astype(int)), 4),
             "최적임계": round(thr2,3), "AUC": round(roc_auc_score(y2[iteb], pt2), 4),
             "LogLoss": round(log_loss(y2[iteb], np.clip(pt2,1e-6,1-1e-6), labels=[0,1]), 4),
             "Brier": round(brier_score_loss(y2[iteb], pt2), 4)})

t = pd.DataFrame(rows)
t.to_csv(OUT / "accuracy_report.csv", index=False, encoding="utf-8-sig")
print(t.to_string(index=False))
print(f"\n2019 base rate = {base:.4f}  →  '항상 아니다' 정확도 = {1-base:.4f}")

# 3) Calibration: 예측확률 십분위별 실제 CSW율
q = pd.qcut(cal_p, 10, labels=False, duplicates="drop")
cal = pd.DataFrame({"pred": cal_p, "actual": cal_y, "bin": q}).groupby("bin").agg(
    예측확률=("pred","mean"), 실제CSW율=("actual","mean"), 건수=("actual","size")).round(4)
cal.to_csv(OUT / "calibration_table.csv", encoding="utf-8-sig")
print("\n[투구 전 최고 모델] 예측확률 십분위별 실제 CSW율")
print(cal.to_string())
