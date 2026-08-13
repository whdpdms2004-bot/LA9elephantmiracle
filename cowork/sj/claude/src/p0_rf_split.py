"""P0-7: game_type(R/F) 분리 모델 vs 단일 모델 실측.

가설 검증 대상
  A single                : 전체 1개 모델 (현행)
  B split                 : R 모델 + F 모델, 추론 시 game_type으로 dispatch
  C split_recentF         : R은 전체, F는 체제 단절 이후 시즌만으로 학습 (F 2022 이전 제외)
  D single_dropOldF       : 단일 모델이지만 학습에서 단절 이전 F 행을 제외
  E single_gtCalib        : 단일 모델 + game_type별 as-of 평균 보정 (직전 시즌 잔차 기반)
  F split_sharedInit      : F 모델을 단일 모델 예측을 base_margin으로 두고 잔차만 학습 (권장 중간안)

평가: 공식 방식대로 pooled 예측 벡터에 대해 BSS. R/F 분해도 함께 기록.
F22/F23/F24 세 fold 모두 측정 (단일 fold 선택 과적합 방지).
"""
import json, os, time
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
gt_raw = df["game_type"].astype(str).copy()
for c in ["top_bottom", "base_state", "game_type"]:
    df[c] = df[c].astype("category").cat.codes
FEATS = sit + asof
y = df[TARGET].values.astype(np.float64)
season = df["season"].values
print("game_type 분포:", gt_raw.value_counts().to_dict(), flush=True)
IS_F = (gt_raw == "F").values

PARAMS = dict(max_depth=0, grow_policy="lossguide", max_leaves=18, eta=0.03,
              subsample=0.8, colsample_bytree=0.6, min_child_weight=64, reg_lambda=2.0,
              objective="binary:logistic", eval_metric="logloss", tree_method="hist",
              nthread=2, seed=0)
ROUNDS = 250


def bss(p, yy):
    r = yy.mean()
    return (1 - np.mean((p - yy) ** 2) / (r * (1 - r))) * 1e5


def train_predict(mask_tr, mask_pr, rounds=ROUNDS, base_margin_tr=None, base_margin_pr=None):
    dtr = xgb.DMatrix(df.loc[mask_tr, FEATS], label=y[mask_tr], missing=np.nan)
    dpr = xgb.DMatrix(df.loc[mask_pr, FEATS], missing=np.nan)
    if base_margin_tr is not None:
        dtr.set_base_margin(base_margin_tr)
        dpr.set_base_margin(base_margin_pr)
    b = xgb.train(PARAMS, dtr, num_boost_round=rounds, verbose_eval=False)
    return b.predict(dpr)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


rows = []
for FOLD in [2022, 2023, 2024]:
    tr = season < FOLD
    va = season == FOLD
    yv = y[va]
    fv = IS_F[va]
    n_f_tr = int((tr & IS_F).sum())
    print(f"\n===== fold {FOLD}: train={tr.sum()} valid={va.sum()} "
          f"F_train={n_f_tr} F_valid={int(fv.sum())} target={yv.mean():.5f}", flush=True)

    P = {}
    # A single
    P["A single"] = train_predict(tr, va)

    # B split
    p = np.empty(va.sum())
    p[~fv] = train_predict(tr & ~IS_F, va & ~IS_F)
    p[fv] = train_predict(tr & IS_F, va & IS_F)
    P["B split"] = p

    # C split_recentF : F는 단절 이후(2023+)만. fold 2022/2023 에서는 직전 1시즌만 사용
    f_start = 2023 if FOLD > 2023 else FOLD - 1
    mF = tr & IS_F & (season >= f_start)
    p = np.empty(va.sum())
    p[~fv] = train_predict(tr & ~IS_F, va & ~IS_F)
    p[fv] = train_predict(mF, va & IS_F)
    P[f"C split_recentF(>={f_start})"] = p

    # D single_dropOldF
    keep = tr & ~(IS_F & (season < f_start))
    P[f"D single_dropOldF(>={f_start})"] = train_predict(keep, va)

    # E single + game_type별 as-of 평균 보정 (직전 시즌의 game_type별 잔차를 그대로 적용)
    prev = season == (FOLD - 1)
    p_prev = train_predict(season < (FOLD - 1), prev)
    shift = {}
    for flag in [False, True]:
        m = (IS_F[prev] == flag)
        if m.sum() > 0:
            shift[flag] = float(np.mean(logit(y[prev][m].mean() * np.ones(1))) -
                                np.mean(logit(p_prev[m])))
    base = P["A single"].copy()
    lg = logit(base)
    for flag in [False, True]:
        m = (fv == flag)
        lg[m] += shift.get(flag, 0.0)
    P["E single + gt as-of shift"] = 1 / (1 + np.exp(-lg))

    # F split_sharedInit : F 행만 단일모델 logit을 base_margin 으로 두고 잔차 부스팅
    bm_tr = logit(train_predict(tr, tr))          # in-sample base margin (진단용; 실제는 OOF 필요)
    p = P["A single"].copy()
    mF_tr = tr & IS_F
    bm_pr = logit(P["A single"][fv])
    resid = train_predict(mF_tr, va & IS_F,
                          base_margin_tr=logit(train_predict(tr, mF_tr)),
                          base_margin_pr=bm_pr)
    p[fv] = resid
    P["F F-residual on single"] = p

    for name, pv in P.items():
        rec = dict(fold=FOLD, config=name,
                   bss=float(bss(pv, yv)), auc=float(roc_auc_score(yv, pv)),
                   pred_mean=float(pv.mean()), target_mean=float(yv.mean()),
                   bss_R=float(bss(pv[~fv], yv[~fv])), bss_F=float(bss(pv[fv], yv[fv])),
                   brier=float(np.mean((pv - yv) ** 2)))
        rows.append(rec)
        print(f"  {name:32s} BSS={rec['bss']:9.2f}  AUC={rec['auc']:.4f}  "
              f"R={rec['bss_R']:9.2f}  F={rec['bss_F']:10.2f}  pred_mean={rec['pred_mean']:.4f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/p0_rf_split.csv", index=False)
print()
print(res.pivot_table(index="config", columns="fold", values="bss").round(1).to_string())
