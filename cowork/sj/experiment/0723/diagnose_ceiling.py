"""예측력 진단: 투구 전 정보의 이론적 한계는 어디인가?

비교:
  (1) pre-pitch  : 지금 우리 모델 (투구 전 정보만)
  (2) +location  : 현재 투구의 '위치'만 추가 (plate_x/z, sz_top/bot)
  (3) +stuff     : 위치 + 구속/무브/회전/구종 (= 오라클, 실제로는 예측 시점에 알 수 없음)
목적: 신호가 '투구 전 맥락'에 있는지 '그 공의 실행'에 있는지 정량화.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import csw_pipeline as P
from lightgbm import LGBMClassifier

CAP = 60_000
OUT = HERE / "out" / "v2"; OUT.mkdir(parents=True, exist_ok=True)

meta = json.loads((HERE / "cache/meta.json").read_text())
feat_df = pd.read_parquet(HERE / "cache/features.parquet")
keep_pitchers = set(feat_df["pitcher"].unique().tolist())

COLS = ["game_year","game_type","pitcher","game_date","game_pk","at_bat_number","pitch_number",
        "description","type","balls","strikes","stand","p_throws","pitch_type",
        "plate_x","plate_z","sz_top","sz_bot","zone",
        "release_speed","pfx_x","pfx_z","release_spin_rate","release_extension"]

pf = pq.ParquetFile(P.RAW)
parts = []
for b in pf.iter_batches(batch_size=250_000, columns=COLS):
    d = b.to_pandas()
    d = d[d["game_type"].eq("R") & d["pitcher"].isin(keep_pitchers)]
    if len(d): parts.append(d)
raw = pd.concat(parts, ignore_index=True)
raw = raw.sort_values(P.SEQ, kind="stable").reset_index(drop=True)
raw = P.add_labels(raw)
y = raw["is_csw"].to_numpy()
tr = raw["game_year"].isin(P.TRAIN_YEARS).to_numpy(); te = raw["game_year"].eq(2019).to_numpy()
tr_pos, te_pos = np.where(tr)[0], np.where(te)[0]
rng = np.random.default_rng(0)
fit = rng.choice(tr_pos, min(CAP, len(tr_pos)), replace=False)

def run(name, cols):
    X = raw[cols].copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                       subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1)
    m.fit(X.iloc[fit], y[fit])
    r = P.metrics(y[te], m.predict_proba(X.iloc[te_pos])[:, 1])
    print(f"{name:24s} logloss {r['logloss']:.4f} | AUC {r['roc_auc']:.4f} | PR {r['pr_auc']:.4f}")
    return r

ctx = ["balls","strikes","stand","p_throws"]
res = {}
res["1_context_only"]      = run("(1) 카운트+좌우",        ctx)
res["2_plus_location"]     = run("(2) +현재 위치",          ctx + ["plate_x","plate_z","sz_top","sz_bot"])
res["3_plus_stuff"]        = run("(3) +구속/무브/구종",     ctx + ["plate_x","plate_z","sz_top","sz_bot",
                                  "release_speed","pfx_x","pfx_z","release_spin_rate","release_extension","pitch_type"])
res["baselines"] = P.baselines(raw, tr, te)
(OUT / "ceiling_diagnosis.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
print("\nbaselines:", {k: v["logloss"] for k, v in res["baselines"].items()})
print("saved out/v2/ceiling_diagnosis.json")
