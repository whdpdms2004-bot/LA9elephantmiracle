"""V85: 두 아이디어를 잰다.

아이디어 A — 행마다 샤프닝/패딩
    예측력이 높은 행은 더 날카롭게, 낮은 행은 기저율 쪽으로 뭉갠다.
        p' = sigmoid( a(x) * (logit(p) - logit(p0)) + logit(p0) )
    a>1 이 샤프닝, a<1 이 패딩이다. 이 프로젝트의 구간별 결합 가중치
    (Public +15) 가 이미 이 아이디어의 특수한 형태다.

    먼저 물어야 할 것: 어떤 행에서 모델이 과신하는가.
    구간별 '보정 기울기' 를 잰다 — logit(y) ~ b * logit(p) 의 b.
        b < 1 이면 과신(패딩이 이득), b > 1 이면 과소(샤프닝이 이득), b = 1 이면 할 게 없다.
    그리고 구간별 최적 a 를 직접 풀어 Brier 이득의 천장을 본다.

아이디어 B — 투수 평균 제구 성공률을 명시적으로
    asof_pitcher_success_rate 는 이미 피처로 들어가 있고 smoothed_10/50/200/500 까지
    있다. 그래도 '명시적' 이 아닌 것은 사실이다 — 모델이 주효과를 스스로 다시
    찾아야 한다. base_margin 으로 주면 잔차만 배운다.

    재학습 전에 값이 있는지부터 본다: 현재 예측의 잔차가 투수 수준 구조를
    아직 갖고 있는가. 남아 있으면 명시화가 이득이고, 없으면 이미 다 쓴 것이다.
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

ARM, FOLDS = "F1", (2023, 2024)      # 2022 는 오염 의심이라 뺀다
PRED = (CAMPAIGN / "outputs" / "single_xgb" /
        "confirm_xgboost_v2r200_tm500_robust_cuda_efull_s20260818_{a}_{f}.npy")

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy()

BINS = {
    "asof_pitcher_n": np.digitize(num("asof_pitcher_n"), [100, 500, 2000, 4000]),
    "count": np.char.add(np.char.add(num("balls_before").astype(int).astype(str), "-"),
                         num("strikes_before").astype(int).astype(str)),
    "asof_batter_n": np.digitize(num("asof_batter_n"), [50, 200, 800, 2000]),
    "li": np.digitize(np.nan_to_num(num("li"), nan=1.0), [0.5, 1.0, 1.5, 2.0, 3.0]),
}

print("=" * 96)
print("A. 어디서 과신하는가 — 구간별 보정 기울기 b  (b<1 과신=패딩, b>1 과소=샤프닝)")
print("=" * 96)
for f in FOLDS:
    va = season == f
    p = np.clip(np.load(str(PRED).format(a=ARM, f=f)).astype(np.float64), EPS, 1 - EPS)
    y = y_all[va]
    z0 = float(np.mean(logit(p)))
    print(f"{chr(10)}  fold {f}   전체 BSS {metrics(y, p)['bss_raw']:.2f}")
    for name, vals in BINS.items():
        g = vals[va]
        print(f"    [{name}]")
        print(f"      {'구간':<10}{'n':>9}{'y평균':>9}{'p평균':>9}"
              f"{'기울기 b':>10}{'최적 a':>9}{'Brier이득':>11}")
        for u in np.unique(g):
            m = g == u
            if m.sum() < 2000:
                continue
            zz, yy = logit(p[m]) - z0, y[m]
            # b: 로지스틱 회귀 기울기 (절편 고정, 뉴턴 2스텝)
            b = 1.0
            for _ in range(30):
                q = sigmoid(b * zz + z0)
                gr = float(np.dot(zz, yy - q))
                he = float(np.dot(zz * zz, q * (1 - q)))
                if he < 1e-9:
                    break
                b += gr / he
            # 최적 a 를 Brier 로 직접 탐색 (b 와 다를 수 있다)
            aa = np.linspace(0.4, 1.8, 71)
            br = [np.mean((sigmoid(a * zz + z0) - yy) ** 2) for a in aa]
            ai = int(np.argmin(br))
            base = np.mean((p[m] - yy) ** 2)
            gain = (base - br[ai]) / (y.mean() * (1 - y.mean())) * 100000
            print(f"      {str(u):<10}{m.sum():>9,}{yy.mean():>9.4f}{p[m].mean():>9.4f}"
                  f"{b:>10.3f}{aa[ai]:>9.2f}{gain:>11.2f}")

print(f"{chr(10)}{'='*96}")
print("B. 잔차에 투수 수준 구조가 남아 있는가")
print("=" * 96)
print(f"  {'fold':>6}{'투수 수':>9}{'잔차 투수분산':>14}{'기대(잡음)':>12}"
      f"{'초과':>10}{'설명가능 BSS':>14}")
pid = df["pitcher_id"].to_numpy()
for f in FOLDS:
    va = season == f
    p = np.clip(np.load(str(PRED).format(a=ARM, f=f)).astype(np.float64), EPS, 1 - EPS)
    y, g = y_all[va], pid[va]
    r = y - p
    t = pd.DataFrame({"g": g, "r": r, "v": p * (1 - p)}).groupby("g").agg(
        n=("r", "size"), m=("r", "mean"), v=("v", "mean"))
    t = t[t.n >= 50]
    obs = float(np.average(t.m ** 2, weights=t.n))
    exp = float(np.average(t.v / t.n, weights=t.n))          # 잔차 평균의 잡음 분산
    ex = obs - exp
    # 초과분을 완전히 제거하면 얻는 BSS (오라클)
    bss = ex / (y.mean() * (1 - y.mean())) * 100000
    print(f"  {f:>6}{len(t):>9,}{obs:>14.6f}{exp:>12.6f}{ex:>+10.6f}{bss:>14.2f}")
print(f"{chr(10)}  '초과' 가 0 이면 투수 수준 정보를 이미 다 쓴 것이다.")
print(f"  양수이면 그만큼이 명시적 투수 사전확률로 회수 가능한 상한이다.")
