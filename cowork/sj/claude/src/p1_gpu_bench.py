"""P1-1: 로컬 GPU 실험 환경 검증 + CPU/GPU 속도 실측 + Val2024 기준선.

이후 모든 실험의 wall-clock 예산과 baseline BSS를 여기서 고정한다.
피처는 원본 48개(row_id 제외)만 사용한다 — 파생/TrackMan 없음.

출력: outputs/p1_gpu_bench.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data" / "train.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
TARGET = "control_success"
SEED = 20260814

df = pd.read_csv(DATA)
for c in ["top_bottom", "game_type", "base_state"]:
    df[c] = df[c].astype("category").cat.codes
FEATS = [c for c in df.columns if c not in ("row_id", TARGET)]
y = df[TARGET].to_numpy(np.float64)
season = df["season"].to_numpy()
X = df[FEATS].to_numpy(np.float32)
del df

tr = season < 2024
va = season == 2024
print(f"train {tr.sum():,}  val {va.sum():,}  feats {len(FEATS)}", flush=True)
dtr = xgb.DMatrix(X[tr], label=y[tr], feature_names=FEATS)
dva = xgb.DMatrix(X[va], label=y[va], feature_names=FEATS)
y_va = y[va]
null_brier = y_va.mean() * (1 - y_va.mean())


def metrics(p):
    brier = float(np.mean((p - y_va) ** 2))
    return brier, 100000 * (1 - brier / null_brier), float(p.mean())


BASE = dict(max_depth=0, grow_policy="lossguide", max_leaves=18, eta=0.03,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=64,
            reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", max_bin=256, seed=SEED)

rows = []
for device, nthread in [("cuda", 24), ("cpu", 24), ("cpu", 6)]:
    params = {**BASE, "device": device, "nthread": nthread}
    t0 = time.perf_counter()
    bst = xgb.train(params, dtr, num_boost_round=800,
                    evals=[(dva, "val")], early_stopping_rounds=60, verbose_eval=200)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    pred_s = time.perf_counter() - t1
    brier, bss, pmean = metrics(p)
    rows.append(dict(model="xgb", device=device, nthread=nthread,
                     best_iter=bst.best_iteration, fit_sec=round(fit_s, 1),
                     pred_sec=round(pred_s, 2), brier=brier, bss=round(bss, 3),
                     pred_mean=round(pmean, 6)))
    print(rows[-1], flush=True)

# CatBoost GPU 가용성
try:
    from catboost import CatBoostClassifier, Pool
    ptr = Pool(X[tr], y[tr])
    pva = Pool(X[va], y[va])
    for task in ["GPU", "CPU"]:
        t0 = time.perf_counter()
        m = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               eval_metric="Logloss", random_seed=SEED,
                               task_type=task, verbose=200,
                               early_stopping_rounds=60)
        m.fit(ptr, eval_set=pva, use_best_model=True)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        p = m.predict_proba(pva)[:, 1]
        pred_s = time.perf_counter() - t1
        brier, bss, pmean = metrics(p)
        rows.append(dict(model="catboost", device=task, nthread=-1,
                         best_iter=m.get_best_iteration(), fit_sec=round(fit_s, 1),
                         pred_sec=round(pred_s, 2), brier=brier, bss=round(bss, 3),
                         pred_mean=round(pmean, 6)))
        print(rows[-1], flush=True)
except Exception as e:  # noqa: BLE001
    print("catboost bench skipped:", repr(e), flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "p1_gpu_bench.csv", index=False)
print("\n" + res.to_string(index=False))
print(f"\nnull_brier(2024) = {null_brier:.8f}  target_mean = {y_va.mean():.8f}")
print(f"saved -> {OUT / 'p1_gpu_bench.csv'}")
