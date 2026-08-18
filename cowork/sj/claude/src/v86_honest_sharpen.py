"""V86: V85 의 두 천장을 정직하게 회수해 본다 (이전 시즌만 사용).

V85 오라클 천장 (fold 2024, 전역 offset 적용 후 기준)
    A 구간별 샤프닝/패딩   count 축 가중합 약 +13.7
    B 투수 잔차 제거       +186.75          <- 이쪽이 훨씬 크다

정직한 버전
    A: 구간별 계수 a 를 season<f 에서 적합해 f 에 적용. 전역 a 쪽으로 축소.
    B: 투수별 잔차를 season<f 에서 EB 수축해 logit 에 더한다.
       lambda = n/(n+K). K 를 넓게 훑어 단조성을 본다.

fold 2022 는 오염 의심이라 확인 fold 에서 뺀다. 다만 학습 재료로는 쓴다.
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

ARM = "F1"
PRED = (CAMPAIGN / "outputs" / "single_xgb" /
        "confirm_xgboost_v2r200_tm500_robust_cuda_efull_s20260818_{a}_{f}.npy")
HAVE = (2022, 2023, 2024)

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
pid = df["pitcher_id"].to_numpy()
num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy()
CNT = np.char.add(np.char.add(num("balls_before").astype(int).astype(str), "-"),
                  num("strikes_before").astype(int).astype(str))

# 전역 offset (all_d100) 을 먼저 걸어 오프셋 교란을 제거한다
def goff(fold):
    tr = season < fold
    r = pd.Series(y_all[tr]).groupby(pd.Series(season[tr])).mean()
    z = np.log(r / (1 - r))
    a, b = np.polyfit(r.index.to_numpy(float), z.to_numpy(), 1)
    return float((a * fold + b) - z.iloc[-1])

Z, Y = {}, {}
for f in HAVE:
    p = np.clip(np.load(str(PRED).format(a=ARM, f=f)).astype(np.float64), EPS, 1 - EPS)
    Z[f] = logit(p) + goff(f)
    Y[f] = y_all[season == f]
BASE = {f: metrics(Y[f], sigmoid(Z[f]))["bss_raw"] for f in HAVE}
print("전역 offset 적용 후 기준선  " + "   ".join(f"{f}={BASE[f]:.2f}" for f in HAVE))

print(f"{chr(10)}{'='*90}")
print("B. 투수별 잔차를 이전 시즌에서 EB 수축해 적용")
print("=" * 90)
print(f"  {'K':>8}{'2023':>12}{'2024':>12}{'2023 Δ':>11}{'2024 Δ':>11}")
for K in [50, 100, 200, 400, 800, 1600, 3200, 10000]:
    out = {}
    for f in (2023, 2024):
        prev = [q for q in HAVE if q < f]
        if not prev:
            continue
        acc = {}
        for q in prev:
            r = Y[q] - sigmoid(Z[q])
            g = pid[season == q]
            t = pd.DataFrame({"g": g, "r": r}).groupby("g")["r"].agg(["size", "mean"])
            for gid, row in t.iterrows():
                s, m = acc.get(gid, (0.0, 0.0))
                acc[gid] = (s + row["size"], m + row["size"] * row["mean"])
        # 잔차(확률 단위)를 로짓 기울기로 환산: dp = p(1-p) dz  ->  dz = dp / (p(1-p))
        gv = pid[season == f]
        p = sigmoid(Z[f])
        dz = np.zeros(len(p))
        sc = p * (1 - p)
        for gid in np.unique(gv):
            if gid not in acc:
                continue
            n, tot = acc[gid]
            if n < 30:
                continue
            m = gv == gid
            dz[m] = (n / (n + K)) * (tot / n) / np.maximum(sc[m], 1e-4)
        dz = np.clip(dz, -1.0, 1.0)
        out[f] = metrics(Y[f], sigmoid(Z[f] + dz))["bss_raw"]
    print(f"  {K:>8}{out[2023]:>12.2f}{out[2024]:>12.2f}"
          f"{out[2023]-BASE[2023]:>+11.2f}{out[2024]-BASE[2024]:>+11.2f}")

print(f"{chr(10)}{'='*90}")
print("A. 볼카운트 구간별 샤프닝 계수 a 를 이전 시즌에서 적합")
print("=" * 90)
print(f"  {'수축 w':>8}{'2023':>12}{'2024':>12}{'2023 Δ':>11}{'2024 Δ':>11}")


def fit_a(zz, yy):
    """logit(y) ~ a*zz 의 a 를 뉴턴법으로."""
    a = 1.0
    for _ in range(40):
        q = sigmoid(a * zz)
        g = float(np.dot(zz, yy - q))
        h = float(np.dot(zz * zz, q * (1 - q)))
        if h < 1e-9:
            break
        step = g / h
        a += step
        if abs(step) < 1e-8:
            break
    return float(np.clip(a, 0.3, 2.0))


for W in [0.0, 0.25, 0.5, 0.75, 1.0]:
    out = {}
    for f in (2023, 2024):
        prev = [q for q in HAVE if q < f]
        z0 = float(np.mean(Z[f]))
        A = {}
        for u in np.unique(CNT):
            zs, ys = [], []
            for q in prev:
                m = CNT[season == q] == u
                if m.sum() >= 2000:
                    zs.append(Z[q][m] - float(np.mean(Z[q])))
                    ys.append(Y[q][m])
            if zs:
                A[u] = fit_a(np.concatenate(zs), np.concatenate(ys))
        zz = Z[f] - z0
        g = CNT[season == f]
        a_row = np.ones(len(zz))
        for u, a in A.items():
            a_row[g == u] = 1.0 + W * (a - 1.0)
        out[f] = metrics(Y[f], sigmoid(a_row * zz + z0))["bss_raw"]
    print(f"  {W:>8.2f}{out[2023]:>12.2f}{out[2024]:>12.2f}"
          f"{out[2023]-BASE[2023]:>+11.2f}{out[2024]-BASE[2024]:>+11.2f}")
print(f"{chr(10)}  W=0 은 아무것도 안 하는 것. 넓은 구간에서 두 fold 양수여야 채택.")
