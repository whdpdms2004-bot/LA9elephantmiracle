"""P0-4: AUC 상한/하한 실측.

질문: fold2024 AUC 0.550이 데이터의 한계인가, 우리 투수 표현의 한계인가?
방법: 피처 블록을 계단식으로 넣어 F24(2019~2023 학습 -> 2024 검증)에서 AUC를 측정.
  S  : 상황 변수만 (카운트/주자/이닝/점수/시즌/월/요일/game_type/손)
  S+A: + asof_* 19개 (운영 제공 과거 이력)
  S+A+ID : + pitcher_id / batter_id 를 범주형으로 (오라클 정체성, 진단용 상한)
  S+A+PTE: + 투수/타자 과거 시즌 target encoding (누수 없는 as-of 근사)
동일 하이퍼파라미터, 동일 시드로 비교.
"""
import json, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

SRC = "/mnt/user-data/uploads/LGAIMERS/data/train.csv"
OUT = "/home/claude/work/outputs"
TARGET = "control_success"

t0 = time.time()
df = pd.read_csv(SRC)
print(f"loaded {df.shape} in {time.time()-t0:.1f}s", flush=True)

# ---- 컬럼 분류
asof = [c for c in df.columns if c.startswith("asof_")]
ids = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]
drop = ["row_id", TARGET]
sit = [c for c in df.columns if c not in asof + ids + drop]
print("situational:", sit)
print("asof:", len(asof))

# ---- 범주형 인코딩
for c in ["top_bottom", "base_state", "game_type"]:
    if c in df.columns:
        df[c] = df[c].astype("category").cat.codes

FOLD = 2024
tr = df["season"] < FOLD
va = df["season"] == FOLD
y_tr, y_va = df.loc[tr, TARGET].values, df.loc[va, TARGET].values
print(f"train={tr.sum()}  valid={va.sum()}  target_mean_va={y_va.mean():.6f}", flush=True)

# ---- as-of 안전한 투수/타자 target encoding (이전 시즌까지의 누적 성공률, smoothing 200)
def asof_te(frame, key, prior_m=200.0):
    g = frame.groupby([key, "season"])[TARGET].agg(["sum", "count"]).reset_index()
    g = g.sort_values([key, "season"])
    g["cum_s"] = g.groupby(key)["sum"].cumsum() - g["sum"]
    g["cum_n"] = g.groupby(key)["count"].cumsum() - g["count"]
    league = frame.groupby("season")[TARGET].mean()
    g["prior"] = g["season"].map(league.shift(1)).fillna(frame[TARGET].mean())
    g["te"] = (g["cum_s"] + prior_m * g["prior"]) / (g["cum_n"] + prior_m)
    g["te_n"] = np.log1p(g["cum_n"])
    return g[[key, "season", "te", "te_n"]].rename(
        columns={"te": f"te_{key}", "te_n": f"ten_{key}"})

for key in ["pitcher_id", "batter_id"]:
    te = asof_te(df, key)
    df = df.merge(te, on=[key, "season"], how="left")
pte = [c for c in df.columns if c.startswith(("te_", "ten_"))]
print("pte cols:", pte, flush=True)

PARAMS = dict(max_depth=0, grow_policy="lossguide", max_leaves=18, eta=0.03,
              subsample=0.8, colsample_bytree=0.6, min_child_weight=64,
              reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss",
              tree_method="hist", nthread=2, seed=0)

BLOCKS = {
    "S  (상황만)":            sit,
    "S+A (+asof)":            sit + asof,
    "S+A+PTE (+as-of TE)":    sit + asof + pte,
    "S+A+ID (오라클 정체성)":  sit + asof + ids,
    "S+A+PTE+ID (전부)":      sit + asof + pte + ids,
}

rows = []
for name, cols in BLOCKS.items():
    cols = [c for c in cols if c in df.columns]
    dtr = xgb.DMatrix(df.loc[tr, cols], label=y_tr, missing=np.nan)
    dva = xgb.DMatrix(df.loc[va, cols], label=y_va, missing=np.nan)
    t = time.time()
    bst = xgb.train(PARAMS, dtr, num_boost_round=1200, evals=[(dva, "va")],
                    early_stopping_rounds=60, verbose_eval=False)
    p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    r = y_va.mean()
    brier = float(np.mean((p - y_va) ** 2))
    nb = brier / (r * (1 - r))
    rec = dict(block=name, n_features=len(cols), best_iter=int(bst.best_iteration),
               auc=float(roc_auc_score(y_va, p)), brier=brier, normalized_brier=nb,
               bss=float((1 - nb) * 1e5), pred_mean=float(p.mean()),
               target_mean=float(r), sec=round(time.time() - t, 1))
    rows.append(rec)
    print(json.dumps(rec, ensure_ascii=False), flush=True)

res = pd.DataFrame(rows)
import os
os.makedirs(OUT, exist_ok=True)
res.to_csv(f"{OUT}/p0_auc_ceiling.csv", index=False)
print()
print(res[["block", "n_features", "auc", "bss", "pred_mean", "best_iter"]].to_string(index=False))
