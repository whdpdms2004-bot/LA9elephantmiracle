"""V87: 투수의 as-of 추세로 행별 offset / 샤프닝을 만든다.

V86 이 실패한 이유
    '이전 시즌 투수 잔차' 를 썼다. 그건 당해 시즌으로 전이되지 않는다(2024 전 K 음수).

이번에 쓰는 것 — 전부 입력 행 안에 있는 as-of 값이다
    수준   asof_pitcher_success_rate            시즌 누적
    추세   asof_pitcher_prev{1,3,5}_game_success_rate - asof_pitcher_success_rate
           (최근 등판이 시즌 평균보다 좋은가 나쁜가)
    신뢰   asof_pitcher_n                        표본 크기
    변동   prev1/3/5 의 표준편차                  기복

    test 행에도 그대로 있으므로 행 독립성·시간 인과 모두 유지된다.
    사상(bin -> a, b)은 season < fold 에서만 적합한다.

모형
    구간마다 Platt 2모수:  logit(y) ~ b_bin + a_bin * (z - z0)
        b_bin  구간 offset  (패딩/이동)
        a_bin  구간 기울기  (>1 샤프닝, <1 패딩)
    수축 W 로 (1, 0) 쪽으로 당긴다. W=0 이면 아무것도 안 하는 것.

fold 2022 는 역신호(V84)라 확인에서 뺀다.
계열별 최선 전역 offset 위에서 잰다 — xgb all_d100, cat last3_d075 (V84).
"""
import os
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

FAMILY = os.environ.get("FAMILY", "xgboost")
ARM, HAVE = "F1", (2022, 2023, 2024)
CFG = {"xgboost": (None, 1.00), "catboost": (3, 0.75)}[FAMILY]
PRED = ((CAMPAIGN / "outputs" / "single_xgb" /
         "confirm_xgboost_v2r200_tm500_robust_cuda_efull_s20260818_{a}_{f}.npy")
        if FAMILY == "xgboost" else
        (CAMPAIGN / "outputs" / "single_catboost" / "{a}_{f}.npy"))

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
n_ = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)

rate = n_("asof_pitcher_success_rate")
prev = np.vstack([n_(f"asof_pitcher_prev{k}_game_success_rate") for k in (1, 3, 5)])
pn = n_("asof_pitcher_n")
trend = np.nanmean(prev, axis=0) - rate          # 최근이 시즌 평균보다 얼마나 좋은가
vol = np.nanstd(prev, axis=0)                    # 기복

# 구간 경계는 고정값 — 데이터에서 분위수를 뽑지 않는다
AX = {
    "추세(prev-누적)": np.digitize(np.nan_to_num(trend), [-0.10, -0.03, 0.03, 0.10]),
    "신뢰(asof_n)": np.digitize(pn, [100, 500, 2000, 4000]),
    "변동(prev sd)": np.digitize(np.nan_to_num(vol, nan=0.0), [0.05, 0.10, 0.16, 0.24]),
    "수준(누적률)": np.digitize(np.nan_to_num(rate, nan=0.5), [0.44, 0.48, 0.52, 0.56]),
}
AX["추세x신뢰"] = AX["추세(prev-누적)"] * 10 + AX["신뢰(asof_n)"]
AX["추세x신뢰x변동"] = AX["추세x신뢰"] * 10 + AX["변동(prev sd)"]


def goff(fold):
    tr = season < fold
    r = pd.Series(y_all[tr]).groupby(pd.Series(season[tr])).mean()
    if CFG[0]:
        r = r.iloc[-CFG[0]:]
    z = np.log(r / (1 - r))
    a, b = np.polyfit(r.index.to_numpy(float), z.to_numpy(), 1)
    return float(CFG[1] * ((a * fold + b) - z.iloc[-1]))


Z, Y = {}, {}
for f in HAVE:
    p = np.clip(np.load(str(PRED).format(a=ARM, f=f)).astype(np.float64), EPS, 1 - EPS)
    Z[f] = logit(p) + goff(f)
    Y[f] = y_all[season == f]
BASE = {f: metrics(Y[f], sigmoid(Z[f]))["bss_raw"] for f in HAVE}
print(f"[{FAMILY}]  전역 offset 적용 후 기준선  "
      + "   ".join(f"{f}={BASE[f]:.2f}" for f in (2023, 2024)))


def platt(z, y):
    """logit(y) ~ b + a*z 를 뉴턴법으로. 반환 (a, b)."""
    a, b = 1.0, 0.0
    for _ in range(50):
        q = sigmoid(a * z + b)
        r = y - q
        w = q * (1 - q)
        g = np.array([np.dot(z, r), r.sum()])
        H = np.array([[np.dot(z * z, w), np.dot(z, w)], [np.dot(z, w), w.sum()]])
        try:
            s = np.linalg.solve(H + np.eye(2) * 1e-6, g)
        except np.linalg.LinAlgError:
            break
        a, b = a + s[0], b + s[1]
        if np.abs(s).max() < 1e-9:
            break
    return float(np.clip(a, 0.3, 2.0)), float(np.clip(b, -0.6, 0.6))


def fit_global(f):
    zs, ys = [], []
    for q in [x for x in HAVE if x < f]:
        zs.append(Z[q] - float(np.mean(Z[q])))
        ys.append(Y[q])
    return platt(np.concatenate(zs), np.concatenate(ys))


print(f"{chr(10)}{'='*92}")
print("적합된 사상 (fold 2024 용, season<2024 에서 적합, 전역 대비 편차)")
print("=" * 92)
f = 2024
z0 = float(np.mean(Z[f]))
for name in ("추세(prev-누적)", "신뢰(asof_n)"):
    vals = AX[name]
    ag, bg = fit_global(f)
    print(f"  [{name}]   전역 a={ag:.3f} b={bg:+.4f}")
    print(f"  {'':<18}{'구간':>6}{'학습 n':>11}{'Δa':>10}{'Δb':>10}")
    for u in np.unique(vals[season == f]):
        zs, ys = [], []
        for q in [x for x in HAVE if x < f]:
            m = vals[season == q] == u
            if m.sum() >= 1000:
                zs.append(Z[q][m] - float(np.mean(Z[q])))
                ys.append(Y[q][m])
        if not zs:
            continue
        a, b = platt(np.concatenate(zs), np.concatenate(ys))
        print(f"  {'':<18}{u:>6}{len(np.concatenate(zs)):>11,}"
              f"{a-ag:>+10.3f}{b-bg:>+10.4f}")

print(f"{chr(10)}{'='*92}")
print("수축 W 를 키우며 — W=0 은 아무것도 안 함")
print("=" * 92)
rows = []
for name, vals in AX.items():
    print(f"{chr(10)}  [{name}]")
    print(f"    {'W':>6}{'2023':>11}{'2024':>11}{'2023 Δ':>10}{'2024 Δ':>10}{'최악 Δ':>10}")
    for W in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        out = {}
        for f in (2023, 2024):
            z0 = float(np.mean(Z[f]))
            zz = Z[f] - z0
            g = vals[season == f]
            ar, br = np.ones(len(zz)), np.zeros(len(zz))
            ag, bg = fit_global(f)
            for u in np.unique(g):
                zs, ys = [], []
                for q in [x for x in HAVE if x < f]:
                    m = vals[season == q] == u
                    if m.sum() >= 1000:
                        zs.append(Z[q][m] - float(np.mean(Z[q])))
                        ys.append(Y[q][m])
                if not zs:
                    continue
                a, b = platt(np.concatenate(zs), np.concatenate(ys))
                sel = g == u
                ar[sel] = 1.0 + W * (a - ag)
                br[sel] = W * (b - bg)
            out[f] = metrics(Y[f], sigmoid(ar * zz + br + z0))["bss_raw"]
        d23, d24 = out[2023] - BASE[2023], out[2024] - BASE[2024]
        print(f"    {W:>6.2f}{out[2023]:>11.2f}{out[2024]:>11.2f}"
              f"{d23:>+10.2f}{d24:>+10.2f}{min(d23, d24):>+10.2f}")
        rows.append({"family": FAMILY, "axis": name, "W": W,
                     "2023": out[2023], "2024": out[2024], "worst_d": min(d23, d24)})
pd.DataFrame(rows).to_csv(
    CAMPAIGN / "outputs" / "combined" / f"v87_pitcher_trend_{FAMILY}.csv", index=False)
print(f"{chr(10)}  채택: 넓은 W 구간에서 두 fold 모두 양수여야 한다.")
