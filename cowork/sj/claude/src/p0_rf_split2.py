"""P0-7b: R/F 분리의 이득 원천 분해 + 미검증 하이브리드 2종.

1차 결과 핵심: R 전용 모델(F 행 제외)은 R 서브셋에서 일관되게 이득
  F22 R 462.69 -> 478.85 (+16.2) | F23 595.83 -> 591.10 (-4.7) | F24 595.78 -> 623.18 (+27.4)
문제는 F 쪽. F 전용 모델은 표본이 적고 2023 체제 단절에서 붕괴한다.

추가 검증
  G hybrid   : R 행 -> R 전용 모델 / F 행 -> 단일 모델(전체 학습)   <- F 표본 고갈 회피
  H hybrid+  : R 행 -> R 전용 모델 / F 행 -> 단일 모델 + F 평균을 직전 시즌 기준으로 강하게 축소 보정
  I weighted : 단일 모델이지만 F 행에 sample_weight w 부여 (w=0.5/2.0)
  J R_only_all: R 전용 모델을 R·F 양쪽에 그대로 적용 (F 전용 처리 없음)
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

SRC = "/mnt/user-data/uploads/LGAIMERS/data/train.csv"
OUT = "/home/claude/work/outputs"
os.makedirs(OUT, exist_ok=True)
TARGET = "control_success"

df = pd.read_csv(SRC)
asof = [c for c in df.columns if c.startswith("asof_")]
ids = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]
sit = [c for c in df.columns if c not in asof + ids + ["row_id", TARGET]]
gt = df["game_type"].astype(str).copy()
for c in ["top_bottom", "base_state", "game_type"]:
    df[c] = df[c].astype("category").cat.codes
FEATS = sit + asof
y = df[TARGET].values.astype(np.float64)
season = df["season"].values
IS_F = (gt == "F").values

PARAMS = dict(max_depth=0, grow_policy="lossguide", max_leaves=18, eta=0.03,
              subsample=0.8, colsample_bytree=0.6, min_child_weight=64, reg_lambda=2.0,
              objective="binary:logistic", eval_metric="logloss", tree_method="hist",
              nthread=2, seed=0)
R = 250


def bss(p, yy):
    r = yy.mean()
    return (1 - np.mean((p - yy) ** 2) / (r * (1 - r))) * 1e5


def tp(mask_tr, mask_pr, w=None):
    dtr = xgb.DMatrix(df.loc[mask_tr, FEATS], label=y[mask_tr], missing=np.nan,
                      weight=None if w is None else w[mask_tr])
    dpr = xgb.DMatrix(df.loc[mask_pr, FEATS], missing=np.nan)
    b = xgb.train(PARAMS, dtr, num_boost_round=R, verbose_eval=False)
    return b.predict(dpr)


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


rows = []
for FOLD in [2022, 2023, 2024]:
    tr, va = season < FOLD, season == FOLD
    yv, fv = y[va], IS_F[va]
    print(f"\n===== fold {FOLD}  target={yv.mean():.5f}  F_valid={int(fv.sum())}", flush=True)
    P = {}
    p_single_va = tp(tr, va)
    P["A single (기준)"] = p_single_va
    p_R_va = tp(tr & ~IS_F, va)              # R 전용 모델을 검증 전체에 적용
    P["J R-only model, 전체 적용"] = p_R_va

    # G hybrid: R행->R모델, F행->단일모델
    p = p_single_va.copy(); p[~fv] = p_R_va[~fv]
    P["G hybrid (R:R모델 / F:단일)"] = p

    # H hybrid + F 축소 보정: 직전 시즌 F 잔차 logit shift를 0.25배로 축소
    prev = season == (FOLD - 1)
    p_prev = tp(season < (FOLD - 1), prev)
    mF = IS_F[prev]
    if mF.sum() > 0:
        raw = float(np.log(y[prev][mF].mean() / (1 - y[prev][mF].mean())) - np.mean(lg(p_prev[mF])))
    else:
        raw = 0.0
    for shrink in [0.25, 0.5]:
        q = p.copy()
        l = lg(q[fv]) + shrink * raw
        q[fv] = 1 / (1 + np.exp(-l))
        P[f"H hybrid + F shift x{shrink}"] = q

    # I weighted single
    for wf in [0.5, 2.0]:
        w = np.where(IS_F, wf, 1.0)
        P[f"I single, F weight={wf}"] = tp(tr, va, w=w)

    for name, pv in P.items():
        rec = dict(fold=FOLD, config=name, bss=float(bss(pv, yv)),
                   auc=float(roc_auc_score(yv, pv)), pred_mean=float(pv.mean()),
                   bss_R=float(bss(pv[~fv], yv[~fv])), bss_F=float(bss(pv[fv], yv[fv])))
        rows.append(rec)
        print(f"  {name:32s} BSS={rec['bss']:9.2f}  AUC={rec['auc']:.4f}  "
              f"R={rec['bss_R']:9.2f}  F={rec['bss_F']:11.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/p0_rf_split2.csv", index=False)
piv = res.pivot_table(index="config", columns="fold", values="bss")
# 목적함수 J = 0.10*NB22 + 0.25*NB23 + 0.65*NB24  ->  가중 BSS 최대화와 동등
piv["weighted_J_BSS"] = 0.10 * piv[2022] + 0.25 * piv[2023] + 0.65 * piv[2024]
print()
print(piv.round(1).sort_values("weighted_J_BSS", ascending=False).to_string())
piv.to_csv(f"{OUT}/p0_rf_split2_pivot.csv")
