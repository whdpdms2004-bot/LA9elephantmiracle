"""원래 목표(투수 CSW% 비율 예측)에서의 유용성 진단.

투구 단위 확률을 (투수×경기) 단위로 평균 → 실제 CSW%와 비교.
비교 대상: 리그평균 / 투수 직전 CSW 이력(rolling) / 우리 모델.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
import csw_pipeline as P
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score

meta = json.loads((HERE / "cache/meta.json").read_text())
df = pd.read_parquet(HERE / "cache/features.parquet")
X = P.build_matrix(df, meta["derived_feats"])
y = df["is_csw"].to_numpy()
tr = df["game_year"].isin(P.TRAIN_YEARS).to_numpy(); te = df["game_year"].eq(2019).to_numpy()
tr_pos, te_pos = np.where(tr)[0], np.where(te)[0]
rng = np.random.default_rng(0)
fit = rng.choice(tr_pos, min(70_000, len(tr_pos)), replace=False)

m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0, random_state=0)
m.fit(X.iloc[fit], y[fit])
p = m.predict_proba(X.iloc[te_pos])[:, 1]

t = df.iloc[te_pos][["pitcher", "game_pk", "is_csw"]].copy()
t["pred"] = p
t["hist"] = df.iloc[te_pos]["p_csw_rate_l500"].to_numpy()      # 최근 500구 이력(투구 전)
league = float(df.loc[tr, "is_csw"].mean())

g = t.groupby(["pitcher", "game_pk"]).agg(
    actual=("is_csw", "mean"), model=("pred", "mean"), hist=("hist", "mean"), n=("is_csw", "size")).reset_index()
g = g[g["n"] >= 30]                                             # 최소 30구 등판

def ev(name, pred):
    return {"방법": name, "MAE": round(mean_absolute_error(g["actual"], pred), 4),
            "RMSE": round(float(np.sqrt(((g["actual"] - pred) ** 2).mean())), 4),
            "R2": round(r2_score(g["actual"], pred), 4),
            "상관": round(float(np.corrcoef(g["actual"], pred)[0, 1]), 4)}

rows = [ev("리그평균(상수)", np.full(len(g), league)),
        ev("투수 최근500구 이력", g["hist"].fillna(league).to_numpy()),
        ev("우리 모델(투구단위 평균)", g["model"].to_numpy())]
out = pd.DataFrame(rows).set_index("방법")
print(f"등판 수(≥30구): {len(g)} | 실제 CSW% 표준편차: {g['actual'].std():.4f}")
print(out.to_string())
(HERE / "out" / "v2" / "rate_diagnosis.json").write_text(
    json.dumps({"n_outings": int(len(g)), "actual_sd": float(g["actual"].std()),
                "results": rows}, ensure_ascii=False, indent=2))
print("saved out/v2/rate_diagnosis.json")
