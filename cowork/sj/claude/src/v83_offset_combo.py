"""V83: V82 에서 살아남은 축들을 조합한다.

V82 결과 — 전역 offset 대비 최악 fold 개선폭 (모든 K 에서 양수인 축만)
    count       +4.39 ~ +5.26     12셀, 셀당 표본 큼
    base_state  +0.76 ~ +3.19
    outs        +1.28 ~ +2.59
    li(중요도)   +1.52 ~ +2.27
    (month -92, game_type -183, score_diff -4 는 탈락)

조합 방식
    가산: 각 축의 (세그먼트offset - 전역offset) 을 축소해 더한다. 축이 독립이라 가정.
    결합: count x li 처럼 교차 셀을 만들어 하나의 축으로 본다. 표본이 12배 준다.
"""
import sys
from itertools import combinations
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

import os
ARM, FOLDS, DAMP, K = "F1", (2022, 2023, 2024), 1.00, 50_000
FAMILY = os.environ.get("FAMILY", "xgboost")
WIN = os.environ.get("WIN", "")
WINDOW = None if WIN in ("", "all") else int(WIN)
DAMP2 = float(os.environ.get("DAMP", "1.0"))
PRED = ((CAMPAIGN / "outputs" / "single_xgb" /
         "confirm_xgboost_v2r200_tm500_robust_cuda_efull_s20260818_{a}_{f}.npy")
        if FAMILY == "xgboost" else
        (CAMPAIGN / "outputs" / "single_catboost" / "{a}_{f}.npy"))

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy()
cat = lambda a, b: np.char.add(np.char.add(a, "|"), b)

AX = {
    "count": np.char.add(np.char.add(num("balls_before").astype(int).astype(str), "-"),
                         num("strikes_before").astype(int).astype(str)),
    "li": np.digitize(np.nan_to_num(num("li"), nan=1.0), [0.5, 1.0, 1.5, 2.0, 3.0]).astype(str),
    "outs": num("outs_before").astype(int).astype(str),
    "base_state": df["base_state"].astype(str).to_numpy(),
}


def off(rates, fold):
    s = rates[(rates.index < fold) & (rates > 0.001) & (rates < 0.999)]
    if WINDOW is not None:
        s = s.iloc[-WINDOW:]
    if len(s) < 2:
        return None
    z = np.log(s.to_numpy() / (1 - s.to_numpy()))
    a, b = np.polyfit(s.index.to_numpy(float), z, 1)
    return float(DAMP2 * ((a * fold + b) - z[-1]))


P, Y, C, GB = {}, {}, {}, {}
for f in FOLDS:
    tr = season < f
    P[f] = np.clip(np.load(str(PRED).format(a=ARM, f=f)).astype(np.float64), EPS, 1 - EPS)
    Y[f] = y_all[season == f]
    C[f] = off(pd.Series(y_all[tr]).groupby(pd.Series(season[tr])).mean(), f)
    GB[f] = metrics(Y[f], sigmoid(logit(P[f]) + C[f]))["bss_raw"]
gw = min(GB.values())


def delta(vals, f, k=K):
    """행별 (세그먼트offset - 전역offset) 을 축소해 반환."""
    tr, va = season < f, season == f
    g, gt, yt, st = vals[va], vals[tr], y_all[tr], season[tr]
    d = np.zeros(int(va.sum()))
    for u in np.unique(g):
        sel = gt == u
        n = int(sel.sum())
        cg = off(pd.Series(yt[sel]).groupby(pd.Series(st[sel])).mean(), f)
        if cg is None or n < 200:
            continue
        d[g == u] = (n / (n + k)) * (cg - C[f])
    return d


def score(build):
    row = [metrics(Y[f], sigmoid(logit(P[f]) + C[f] + build(f)))["bss_raw"] for f in FOLDS]
    return row, min(row)


print(f"[{FAMILY}]  window={WIN or 'all'}  damping={DAMP2}")
print(f"전역 offset  " + "  ".join(f"{f}={GB[f]:.2f}" for f in FOLDS) + f"   최악 {gw:.2f}")
print(f"{chr(10)}{'='*96}")
print(f"가산 조합  (K={K:,}, 각 축의 편차를 축소해 더한다)")
print("=" * 96)
print(f"  {'조합':<34}", end="")
for f in FOLDS:
    print(f"{f:>11}", end="")
print(f"{'최악':>10}{'전역대비':>10}")
names = list(AX)
res = []
for r in range(1, len(names) + 1):
    for combo in combinations(names, r):
        row, w = score(lambda f, c=combo: sum(delta(AX[n], f) for n in c))
        tag = " + ".join(combo)
        print(f"  {tag:<34}", end="")
        for b in row:
            print(f"{b:>11.2f}", end="")
        print(f"{w:>10.2f}{w-gw:>+10.2f}")
        res.append({"kind": "add", "combo": tag, "worst": w, "vs_global": w - gw,
                    **{str(f): row[i] for i, f in enumerate(FOLDS)}})

print(f"{chr(10)}{'='*96}")
print("교차 셀  (하나의 축으로 본다 — 표본이 급감한다)")
print("=" * 96)
print(f"  {'축':<34}{'셀':>5}", end="")
for f in FOLDS:
    print(f"{f:>11}", end="")
print(f"{'최악':>10}{'전역대비':>10}")
for combo in [("count", "li"), ("count", "outs"), ("count", "li", "outs")]:
    v = AX[combo[0]]
    for n in combo[1:]:
        v = cat(v, AX[n])
    row, w = score(lambda f, vv=v: delta(vv, f))
    print(f"  {' x '.join(combo):<34}{len(np.unique(v)):>5}", end="")
    for b in row:
        print(f"{b:>11.2f}", end="")
    print(f"{w:>10.2f}{w-gw:>+10.2f}")
    res.append({"kind": "cross", "combo": " x ".join(combo), "worst": w,
                "vs_global": w - gw, **{str(f): row[i] for i, f in enumerate(FOLDS)}})

pd.DataFrame(res).to_csv(CAMPAIGN / "outputs" / "combined" / f"v83_offset_combo_{FAMILY}_{WIN or 'all'}_d{int(DAMP2*100):03d}.csv", index=False)
b = max(res, key=lambda r: r["vs_global"])
print(f"{chr(10)}  최고: {b['combo']} ({b['kind']})   최악 fold {b['worst']:.2f}  "
      f"전역대비 {b['vs_global']:+.2f}")
print(f"  주의: fold 2022 는 오염 의심이라 '최악'이 2023 인지 확인할 것.")
