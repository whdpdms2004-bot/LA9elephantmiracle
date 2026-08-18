"""V82: 경기 중요도 / 볼카운트 / 월 별 offset.

V80·V81 에서 축 이름을 틀려(month/balls/strikes) 세 축이 통째로 빠져 있었다.
실제 컬럼은 game_month, balls_before, strikes_before, 그리고 li(leverage index).
li 가 이 데이터의 '경기 중요도' 지표다.

카운트는 12셀이고 셀당 표본이 커서 game_type(F가 작다)보다 추정 조건이 훨씬 낫다.
V81 의 실패가 축의 문제인지 추정량의 문제인지 여기서 갈린다.

두 단계
    1. 오라클 천장 — 각 세그먼트를 실제 평균으로 재중심화. 이 아이디어의 상한.
    2. 정직한 축소 — 세그먼트 offset 을 전역 쪽으로 lambda_g = n_g/(n_g+K) 로 축소.
       K=inf 가 전역 방식과 완전히 동일하도록 겹쳐 놓았다.
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
KS = [0, 5_000, 20_000, 50_000, 150_000, 500_000, np.inf]

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy()

# li 구간은 리버리지 인덱스 관례값 고정 — 데이터에서 분위수를 뽑지 않는다
AX = {
    "count":      np.char.add(np.char.add(num("balls_before").astype(int).astype(str),
                                          "-"),
                              num("strikes_before").astype(int).astype(str)),
    "month":      num("game_month").astype(int).astype(str),
    "li(중요도)":  np.digitize(np.nan_to_num(num("li"), nan=1.0),
                              [0.5, 1.0, 1.5, 2.0, 3.0]).astype(str),
    "score_diff": np.digitize(np.nan_to_num(num("score_diff_pitcher_team"), nan=0),
                              [-4, -2, -1, 0, 1, 2, 4]).astype(str),
    "outs":       num("outs_before").astype(int).astype(str),
    "base_state": df["base_state"].astype(str).to_numpy(),
    "game_type":  df["game_type"].astype(str).to_numpy(),
}


def off(rates, fold, damping=DAMP):
    s = rates[(rates.index < fold) & (rates > 0.001) & (rates < 0.999)]
    if len(s) < 2:
        return None
    z = np.log(s.to_numpy() / (1 - s.to_numpy()))
    a, b = np.polyfit(s.index.to_numpy(float), z, 1)
    return float(damping * ((a * fold + b) - z[-1]))


def shift_to(p, target):
    z = logit(np.clip(p, EPS, 1 - EPS))
    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(z + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return np.clip(sigmoid(z + (lo + hi) / 2), EPS, 1 - EPS)


P, Y, BASE, C, GB = {}, {}, {}, {}, {}
for f in FOLDS:
    tr = season < f
    P[f] = np.clip(np.load(str(PRED).format(a=ARM, f=f)).astype(np.float64), EPS, 1 - EPS)
    Y[f] = y_all[season == f]
    BASE[f] = metrics(Y[f], P[f])
    C[f] = off(pd.Series(y_all[tr]).groupby(pd.Series(season[tr])).mean(), f)
    GB[f] = metrics(Y[f], sigmoid(logit(P[f]) + C[f]))["bss_raw"]
gw = min(GB.values())
print(f"전역 offset  " + "  ".join(f"{f}={GB[f]:.2f}" for f in FOLDS) + f"   최악 {gw:.2f}")
print(f"전역 재중심화(오라클)  " + "  ".join(f"{f}={BASE[f]['bss_centered']:.2f}" for f in FOLDS))

print(f"{chr(10)}{'='*98}")
print("1. 오라클 천장 — 각 세그먼트를 실제 평균으로 재중심화 (전역 재중심화 대비)")
print("=" * 98)
print(f"  {'축':<12}{'셀':>5}{'최소셀n':>9}", end="")
for f in FOLDS:
    print(f"{f:>11}", end="")
print(f"{'평균이득':>10}")
for name, vals in AX.items():
    row = []
    for f in FOLDS:
        va = season == f
        p, y, g = P[f], Y[f], vals[va]
        q = p.copy()
        for u in np.unique(g):
            m = g == u
            if m.sum() >= 30:
                q[m] = shift_to(p[m], float(y[m].mean()))
        row.append(metrics(y, q)["bss_raw"])
    gain = np.mean([row[i] - BASE[f]["bss_centered"] for i, f in enumerate(FOLDS)])
    mn = min(int((vals[season == f] == u).sum()) for f in FOLDS for u in np.unique(vals))
    print(f"  {name:<12}{len(np.unique(vals)):>5}{mn:>9,}", end="")
    for b in row:
        print(f"{b:>11.2f}", end="")
    print(f"{gain:>+10.2f}")

print(f"{chr(10)}{'='*98}")
print("2. 정직한 축소  ·  K=inf 는 전역 방식과 완전히 동일")
print("=" * 98)
rows = []
for name, vals in AX.items():
    best = None
    print(f"{chr(10)}  [{name}]")
    print(f"    {'K':>9}", end="")
    for f in FOLDS:
        print(f"{f:>11}", end="")
    print(f"{'최악':>10}{'전역대비':>10}")
    for K in KS:
        row = []
        for f in FOLDS:
            tr, va = season < f, season == f
            z = logit(P[f]).copy()
            g, gt, yt, st = vals[va], vals[tr], y_all[tr], season[tr]
            for u in np.unique(g):
                m = g == u
                sel = gt == u
                cg = off(pd.Series(yt[sel]).groupby(pd.Series(st[sel])).mean(), f)
                if cg is None or int(sel.sum()) < 200:
                    c = C[f]
                else:
                    lam = 0.0 if K == np.inf else (1.0 if K == 0 else int(sel.sum()) / (int(sel.sum()) + K))
                    c = C[f] + lam * (cg - C[f])
                z[m] += c
            row.append(metrics(Y[f], sigmoid(z))["bss_raw"])
        w = min(row)
        kt = "inf" if K == np.inf else f"{K:,}"
        print(f"    {kt:>9}", end="")
        for b in row:
            print(f"{b:>11.2f}", end="")
        print(f"{w:>10.2f}{w-gw:>+10.2f}")
        rows.append({"axis": name, "K": kt, **{str(f): row[i] for i, f in enumerate(FOLDS)},
                     "worst": w, "vs_global": w - gw})
pd.DataFrame(rows).to_csv(CAMPAIGN / "outputs" / "combined" / "v82_offset_axes.csv", index=False)
print(f"{chr(10)}  판정: 넓은 K 구간에서 '전역대비'가 양수여야 채택.")
