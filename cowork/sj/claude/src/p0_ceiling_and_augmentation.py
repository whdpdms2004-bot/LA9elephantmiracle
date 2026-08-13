"""P0-4c: 표현 granularity별 판별력 상한 (모델 적합 없이 정확 측정) + P0-6: 데이터 증강 효과 점검.

[1] Oracle ceiling — LOO 그룹 평균은 그 자체가 확률이므로 XGB 없이 AUC/BSS를 직접 측정한다.
    (1차 실험은 dominant feature 때문에 26 iter에서 조기 종료 -> underfit 아티팩트였음)
    이 값은 "그 granularity의 완벽한 표현이 도달할 수 있는 최대치"다.

[2] 증분 측정 — 조기 종료를 끄고 고정 라운드로 BASE vs BASE+oracle 비교.

[3] 학습곡선 — 학습 행을 12.5 / 25 / 50 / 100%로 줄여 AUC/BSS 측정.
    곡선이 평평하면 표본 수가 병목이 아니므로 데이터 증강은 무의미하다.

[4] 좌우 미러 증강 타당성 — 좌투/우투 표본 불균형 확인.
"""
import json, os, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

SRC = "/mnt/user-data/uploads/LGAIMERS/data/train.csv"
OUT = "/home/claude/work/outputs"
os.makedirs(OUT, exist_ok=True)
TARGET, FOLD = "control_success", 2024

df = pd.read_csv(SRC)
asof = [c for c in df.columns if c.startswith("asof_")]
ids = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]
sit = [c for c in df.columns if c not in asof + ids + ["row_id", TARGET]]
for c in ["top_bottom", "base_state", "game_type"]:
    df[c] = df[c].astype("category").cat.codes
y = df[TARGET].values.astype(np.float64)
va = (df["season"] == FOLD).values
tr = (df["season"] < FOLD).values
y_va, y_tr = y[va], y[tr]
r = y_va.mean()
V = r * (1 - r)


def bss(p, yy=y_va):
    rr = yy.mean()
    return (1 - np.mean((p - yy) ** 2) / (rr * (1 - rr))) * 1e5


def loo(keys, m=50.0):
    g = df.groupby(keys)[TARGET]
    s = g.transform("sum").values
    n = g.transform("count").values
    league = df.groupby("season")[TARGET].transform("mean").values
    return ((s - y) + m * league) / ((n - 1) + m), n


GRAN = {
    "O1 (pitcher,season)": ["pitcher_id", "season"],
    "O2 (pitcher,season,bhand)": ["pitcher_id", "season", "batter_hand"],
    "O3 (pitcher,season,count)": ["pitcher_id", "season", "balls_before", "strikes_before"],
    "O3b (pitcher,season,count,bhand)": ["pitcher_id", "season", "balls_before", "strikes_before", "batter_hand"],
    "O4 (pitcher,batter,season)": ["pitcher_id", "batter_id", "season"],
    "O6 (batter,season)": ["batter_id", "season"],
    "O7 (pitcher,season,inning)": ["pitcher_id", "season", "inning"],
}

print("=" * 90)
print("[1] Oracle ceiling — 해당 granularity의 완벽한 표현이 낼 수 있는 최대 AUC/BSS (2024 검증)")
print("=" * 90)
rows = []
for name, keys in GRAN.items():
    p, n = loo(keys)
    pv = p[va]
    rows.append(dict(granularity=name, auc=float(roc_auc_score(y_va, pv)), bss=float(bss(pv)),
                     pred_mean=float(pv.mean()), median_cell_n=float(np.median(n[va])),
                     cells=int(df.loc[va].groupby(keys).ngroups)))
    print(f"  {name:36s} AUC={rows[-1]['auc']:.4f}  BSS={rows[-1]['bss']:9.1f}  "
          f"cells={rows[-1]['cells']:6d}  median n={rows[-1]['median_cell_n']:.0f}")
pd.DataFrame(rows).to_csv(f"{OUT}/p0_oracle_standalone.csv", index=False)

PARAMS = dict(max_depth=0, grow_policy="lossguide", max_leaves=18, eta=0.03,
              subsample=0.8, colsample_bytree=0.6, min_child_weight=64, reg_lambda=2.0,
              objective="binary:logistic", eval_metric="logloss", tree_method="hist",
              nthread=2, seed=0)
BASE = sit + asof


def fit(cols, mask_tr, rounds=250):
    dtr = xgb.DMatrix(df.loc[mask_tr, cols], label=y[mask_tr], missing=np.nan)
    dva = xgb.DMatrix(df.loc[va, cols], label=y_va, missing=np.nan)
    b = xgb.train(PARAMS, dtr, num_boost_round=rounds, verbose_eval=False)
    p = b.predict(dva)
    return float(roc_auc_score(y_va, p)), float(bss(p)), float(p.mean())


print()
print("=" * 90)
print("[2] 증분 — 조기 종료 없이 고정 250 라운드")
print("=" * 90)
inc = []
a, s, pm = fit(BASE, tr)
inc.append(dict(block="BASE (S+A)", auc=a, bss=s, pred_mean=pm))
print(f"  {'BASE (S+A)':36s} AUC={a:.4f}  BSS={s:9.1f}  pred_mean={pm:.4f}")
for name, keys in GRAN.items():
    p, n = loo(keys)
    col = "orc_" + name.split()[0]
    df[col] = p
    df[col + "_n"] = np.log1p(n - 1)
    a, s, pm = fit(BASE + [col, col + "_n"], tr)
    inc.append(dict(block="BASE + " + name, auc=a, bss=s, pred_mean=pm))
    print(f"  {'BASE + ' + name:36s} AUC={a:.4f}  BSS={s:9.1f}  pred_mean={pm:.4f}")
pd.DataFrame(inc).to_csv(f"{OUT}/p0_oracle_incremental.csv", index=False)

print()
print("=" * 90)
print("[3] 학습곡선 — 표본 수가 병목인가? (평평하면 데이터 증강 무의미)")
print("=" * 90)
rng = np.random.default_rng(0)
idx_tr = np.flatnonzero(tr)
lc = []
for frac in [0.125, 0.25, 0.5, 1.0]:
    k = int(len(idx_tr) * frac)
    sub = np.zeros(len(df), bool)
    sub[rng.choice(idx_tr, k, replace=False)] = True
    a, s, pm = fit(BASE, sub)
    lc.append(dict(frac=frac, n_train=k, auc=a, bss=s))
    print(f"  train {frac*100:5.1f}%  n={k:8d}  AUC={a:.4f}  BSS={s:9.1f}")
# 최근 시즌만 (표본 절반이지만 분포가 가까움) - '양 vs 최근성' 비교
recent = tr & df["season"].ge(2022).values
a, s, pm = fit(BASE, recent)
lc.append(dict(frac=float(recent.sum() / len(idx_tr)), n_train=int(recent.sum()),
               auc=a, bss=s, note="2022~2023만"))
print(f"  2022~2023만  n={int(recent.sum()):8d}  AUC={a:.4f}  BSS={s:9.1f}   <- 양보다 최근성 비교")
pd.DataFrame(lc).to_csv(f"{OUT}/p0_learning_curve.csv", index=False)

print()
print("=" * 90)
print("[4] 좌우 미러 증강 타당성 — 손 조합별 표본")
print("=" * 90)
t = df.groupby(["pitcher_hand", "batter_hand"]).agg(n=(TARGET, "size"), rate=(TARGET, "mean"))
t["share"] = t["n"] / len(df)
print(t.to_string())
ph = df.groupby("pitcher_hand")["pitcher_id"].nunique()
print("\n투수 수 (hand별):", ph.to_dict())
print("행 비중 (hand별):", (df.groupby('pitcher_hand').size() / len(df)).round(4).to_dict())
