"""V80: offset 을 세그먼트별로 쪼개면 fold 격차가 줄어드는가.

배경
    evaluate_train_only_season_offsets 가 이전 시즌만으로 정한 전역 logit offset 을
    걸어 세 fold 를 전부 양수로 만들었다 (2167.93 / 22.02 / 821.66).
    그런데 fold 간 격차가 여전히 크다. offset 을 fold 를 나눈 기준(시즌) 안쪽의
    세그먼트별로 다르게 주면 격차가 줄어들까?

이 스크립트가 답하는 순서
    1. 격차가 정말 offset 때문인가 — bss_centered(오프셋을 완전히 제거한 BSS)를 본다.
       여기서 이미 격차가 남으면 offset 을 아무리 정교하게 해도 못 줄인다.
    2. 세그먼트 offset 의 천장 — 각 세그먼트를 '실제 평균'으로 재중심화한다(오라클).
       전역 재중심화 대비 얼마나 더 버는지가 이 아이디어 전체의 상한이다.
    3. 정직한 버전 — 세그먼트 offset 을 이전 시즌만으로 추정해 적용한다.
       천장의 몇 %를 회수하는지 본다. V36(사후보정 최악 -162)의 실패 경로를 밟는지.

실행: python v80_segmented_offset.py
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

ARM, FOLDS = "F1", (2022, 2023, 2024)
PRED = (CAMPAIGN / "outputs" / "single_xgb" /
        "confirm_xgboost_v2r200_tm500_robust_cuda_efull_s20260818_{a}_{f}.npy")

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)

# ── 세그먼트 축 정의 (전부 행 단위로 결정된다 = 행 독립성 유지) ──────────────
def seg_axes(d):
    out = {}
    if "month" in d:
        out["month"] = d["month"].astype("Int64").astype(str).to_numpy()
    if "game_type" in d:
        out["game_type"] = d["game_type"].astype(str).to_numpy()
    if {"balls", "strikes"} <= set(d.columns):
        out["count"] = (d["balls"].astype(str) + "-" + d["strikes"].astype(str)).to_numpy()
    if "batter_hand" in d:
        out["batter_hand"] = d["batter_hand"].astype(str).to_numpy()
    if "inning" in d:
        out["inning"] = np.clip(pd.to_numeric(d["inning"], errors="coerce")
                                .fillna(0).to_numpy().astype(int), 1, 10).astype(str)
    if "asof_pitcher_n" in d:
        out["vol_bucket"] = np.digitize(
            d["asof_pitcher_n"].to_numpy(), [100, 500, 2000, 4000]).astype(str)
    return out


AX = seg_axes(df)
print(f"세그먼트 축 {len(AX)}개: {', '.join(AX)}")


def shift_to(p, target):
    """평균이 target 이 되도록 로짓 평행이동 (이분 탐색)."""
    z = logit(np.clip(p, EPS, 1 - EPS))
    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(z + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return np.clip(sigmoid(z + (lo + hi) / 2), EPS, 1 - EPS)


print(f"{chr(10)}{'='*94}")
print("1. 격차가 offset 때문인가 — 오프셋을 완전히 제거해도 남는 것")
print("=" * 94)
print(f"  {'fold':>6}{'n':>10}{'실제평균':>10}{'예측평균':>10}{'오프셋':>10}"
      f"{'bss_raw':>11}{'bss_centered':>14}")
P, Y, M = {}, {}, {}
for f in FOLDS:
    va = season == f
    p = np.clip(np.load(str(PRED).format(a=ARM, f=f)).astype(np.float64), EPS, 1 - EPS)
    y = y_all[va]
    assert len(p) == len(y), f"길이 불일치 {len(p)} vs {len(y)}"
    P[f], Y[f] = p, y
    m = metrics(y, p)
    M[f] = m
    print(f"  {f:>6}{len(y):>10,}{y.mean():>10.4f}{p.mean():>10.4f}"
          f"{m['offset']:>+10.4f}{m['bss_raw']:>11.2f}{m['bss_centered']:>14.2f}")
sp_raw = max(M[f]["bss_raw"] for f in FOLDS) - min(M[f]["bss_raw"] for f in FOLDS)
sp_ctr = max(M[f]["bss_centered"] for f in FOLDS) - min(M[f]["bss_centered"] for f in FOLDS)
print(f"{chr(10)}  fold 격차(최대−최소)   bss_raw {sp_raw:9.1f}   bss_centered {sp_ctr:9.1f}")
print(f"  → 오프셋을 전부 지워도 격차가 {sp_ctr/sp_raw*100:.0f}% 남는다면 "
      f"offset 은 격차의 원인이 아니다.")

print(f"{chr(10)}{'='*94}")
print("2. 세그먼트 offset 의 천장 — 각 세그먼트를 '실제 평균'으로 재중심화 (오라클)")
print("=" * 94)
print(f"  {'축':<14}{'셀':>6}", end="")
for f in FOLDS:
    print(f"{f:>12}", end="")
print(f"{'평균 이득':>12}")
print(f"  {'전역 재중심화':<14}{1:>6}", end="")
for f in FOLDS:
    print(f"{M[f]['bss_centered']:>12.2f}", end="")
print(f"{np.mean([M[f]['bss_centered']-M[f]['bss_raw'] for f in FOLDS]):>12.2f}")

ceil = {}
for name, vals in AX.items():
    row, gains = [], []
    for f in FOLDS:
        va = season == f
        p, y, g = P[f], Y[f], vals[va]
        q = p.copy()
        for u in np.unique(g):
            m = g == u
            if m.sum() >= 30:
                q[m] = shift_to(p[m], float(y[m].mean()))
        b = metrics(y, q)["bss_raw"]
        row.append(b)
        gains.append(b - M[f]["bss_centered"])
    ceil[name] = row
    ncell = len(np.unique(vals))
    print(f"  {name:<14}{ncell:>6}", end="")
    for b in row:
        print(f"{b:>12.2f}", end="")
    print(f"{np.mean(gains):>12.2f}   <- 전역 재중심화 대비")

print(f"{chr(10)}  '평균 이득' 은 전역 재중심화 대비 추가분이다. 이것이 세그먼트화의 상한이며,")
print(f"  실제 값을 보고 맞춘 오라클이라 정직한 방법으로는 이보다 반드시 작다.")

print(f"{chr(10)}{'='*94}")
print("3. 정직한 버전 — 세그먼트 offset 을 이전 시즌만으로 추정")
print("=" * 94)
print(f"  {'축':<14}", end="")
for f in FOLDS:
    print(f"{f:>12}", end="")
print(f"{'최악':>11}{'천장회수율':>12}")

for name, vals in AX.items():
    row = []
    for f in FOLDS:
        tr, va = season < f, season == f
        p, y, g = P[f], Y[f], vals[va]
        gt, yt, st = vals[tr], y_all[tr], season[tr]
        prev = sorted(set(st.tolist()))
        if len(prev) < 2:
            row.append(np.nan)
            continue
        # 세그먼트별 로짓 평균의 시즌 추세를 선형 외삽 -> 전역 추세와의 차이만 사용
        gl = pd.Series(yt).groupby(pd.Series(st)).mean()
        gz = np.log(gl / (1 - gl))
        a_, b_ = np.polyfit(gl.index.to_numpy(float), gz.to_numpy(), 1)
        glob_fc = a_ * f + b_
        q = p.copy()
        for u in np.unique(g):
            m = g == u
            if m.sum() < 30:
                continue
            sel = gt == u
            sr = pd.Series(yt[sel]).groupby(pd.Series(st[sel])).mean()
            sr = sr[(sr > 0.001) & (sr < 0.999)]
            if len(sr) < 2 or sel.sum() < 500:
                tgt = float(1 / (1 + np.exp(-glob_fc)))
            else:
                sz = np.log(sr / (1 - sr))
                a2, b2 = np.polyfit(sr.index.to_numpy(float), sz.to_numpy(), 1)
                tgt = float(1 / (1 + np.exp(-(a2 * f + b2))))
            q[m] = shift_to(p[m], tgt)
        row.append(metrics(y, q)["bss_raw"])
    ceil_gain = np.mean([ceil[name][i] - M[f]["bss_centered"] for i, f in enumerate(FOLDS)])
    hon_gain = np.mean([row[i] - M[f]["bss_centered"] for i, f in enumerate(FOLDS)])
    print(f"  {name:<14}", end="")
    for b in row:
        print(f"{b:>12.2f}", end="")
    rec = hon_gain / ceil_gain * 100 if abs(ceil_gain) > 1e-9 else float("nan")
    print(f"{np.nanmin(row):>11.2f}{rec:>11.0f}%")

print(f"{chr(10)}  판단: 최악 fold 가 전역 재중심화보다 나빠지면 기각이다.")
print(f"  V36(사후보정, 최악 -162)과 같은 경로 — 한 해의 사상이 다른 해에 틀린다.")
