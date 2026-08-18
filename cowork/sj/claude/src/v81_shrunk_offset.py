"""V81: 세그먼트별 logit offset 을 전역 offset 쪽으로 축소하면 살아나는가.

원래 방식(evaluate_train_only_season_offsets)과 정확히 같은 형태를 쓴다.
    offset = damping x (로짓 선형외삽 - 직전시즌 로짓)
    p' = sigmoid(logit(p) + offset)

세그먼트 버전
    c_g = 세그먼트 g 의 시즌별 비율로 같은 계산
    c_g(축소) = c_global + lambda_g x (c_g - c_global),   lambda_g = n_g/(n_g+K)
    lambda=0 (K=inf) 이면 전역 방식과 '완전히 동일'하다. 제대로 겹쳐진 비교다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# 2026-08-18 feature_campaign_1000 -> claude/src 이관.
# 데이터/산출물은 캠페인 폴더에 그대로 있으므로 CAMPAIGN 으로 가리킨다.
CAMPAIGN = HERE.parents[1] / "feature_campaign_1000"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CAMPAIGN))
from evaluate_bucketed_residual import EPS, load, logit, sigmoid
from harness import TARGET, metrics

ARM, FOLDS, DAMP = "F1", (2022, 2023, 2024), 1.00
PRED = (CAMPAIGN / "outputs" / "single_xgb" /
        "confirm_xgboost_v2r200_tm500_robust_cuda_efull_s20260818_{a}_{f}.npy")
KS = [0, 1_000, 5_000, 20_000, 50_000, 150_000, 500_000, np.inf]

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
AX = {
    "game_type": df["game_type"].astype(str).to_numpy(),
    "vol_bucket": np.digitize(df["asof_pitcher_n"].to_numpy(),
                              [100, 500, 2000, 4000]).astype(str),
    "inning": np.clip(pd.to_numeric(df["inning"], errors="coerce").fillna(0)
                      .to_numpy().astype(int), 1, 10).astype(str),
    "batter_hand": df["batter_hand"].astype(str).to_numpy(),
}


def off(rates, fold, damping=DAMP):
    """원 스크립트와 동일: damping x (로짓 선형외삽 - 직전시즌 로짓)."""
    s = rates[(rates.index < fold) & (rates > 0.001) & (rates < 0.999)]
    if len(s) < 2:
        return None
    x = s.index.to_numpy(float)
    z = np.log(s.to_numpy() / (1 - s.to_numpy()))
    a, b = np.polyfit(x, z, 1)
    return float(damping * ((a * fold + b) - z[-1]))


P, Y, BASE, C = {}, {}, {}, {}
for f in FOLDS:
    va, tr = season == f, season < f
    P[f] = np.clip(np.load(str(PRED).format(a=ARM, f=f)).astype(np.float64), EPS, 1 - EPS)
    Y[f] = y_all[va]
    BASE[f] = metrics(Y[f], P[f])
    C[f] = off(pd.Series(y_all[tr]).groupby(pd.Series(season[tr])).mean(), f)

print("=" * 96)
print("기준선  (damping 1.00, 원 스크립트의 all_d100 과 동일)")
print("=" * 96)
print(f"  {'fold':>6}{'c_global':>11}{'보정없음':>12}{'전역 offset':>14}"
      f"{'전역 재중심화(오라클)':>22}")
GB = {}
for f in FOLDS:
    q = sigmoid(logit(P[f]) + C[f])
    GB[f] = metrics(Y[f], q)["bss_raw"]
    print(f"  {f:>6}{C[f]:>+11.4f}{BASE[f]['bss_raw']:>12.2f}{GB[f]:>14.2f}"
          f"{BASE[f]['bss_centered']:>22.2f}")
gw = min(GB.values())
print(f"{chr(10)}  전역 offset 최악 fold = {gw:.2f}   (이 값을 넘어야 채택)")

print(f"{chr(10)}{'='*96}")
print("세그먼트 offset 을 전역 쪽으로 축소  ·  K=inf 는 전역 방식과 완전히 동일")
print("=" * 96)
rows = []
for name, vals in AX.items():
    print(f"{chr(10)}  [{name}]   셀 {len(np.unique(vals))}개")
    print(f"    {'K':>9}", end="")
    for f in FOLDS:
        print(f"{f:>12}", end="")
    print(f"{'최악':>11}{'전역대비':>11}")
    for K in KS:
        row = []
        for f in FOLDS:
            tr, va = season < f, season == f
            p, g, gt, yt, st = P[f], vals[va], vals[tr], y_all[tr], season[tr]
            z = logit(p).copy()
            for u in np.unique(g):
                m = g == u
                sel = gt == u
                n_g = int(sel.sum())
                cg = off(pd.Series(yt[sel]).groupby(pd.Series(st[sel])).mean(), f)
                if cg is None or n_g < 200:
                    c = C[f]
                else:
                    lam = 0.0 if K == np.inf else (1.0 if K == 0 else n_g / (n_g + K))
                    c = C[f] + lam * (cg - C[f])
                z[m] += c
            row.append(metrics(Y[f], sigmoid(z))["bss_raw"])
        worst = min(row)
        kt = "inf" if K == np.inf else f"{K:,}"
        print(f"    {kt:>9}", end="")
        for b in row:
            print(f"{b:>12.2f}", end="")
        print(f"{worst:>11.2f}{worst-gw:>+11.2f}")
        rows.append({"axis": name, "K": kt,
                     **{str(f): row[i] for i, f in enumerate(FOLDS)},
                     "worst": worst, "vs_global": worst - gw})

pd.DataFrame(rows).to_csv(CAMPAIGN / "outputs" / "combined" / "v81_shrunk_offset.csv", index=False)
print(f"{chr(10)}  판정: 넓은 K 구간에서 '전역대비'가 양수여야 채택이다.")
