"""V88: v85_preprocess_screen 의 arm 순위가 전역 offset 선택에 흔들리는가.

문제
    v85_preprocess_screen 은 `forecast_offset(..., window=None, damping=0.25)`
    = all_d025 로 고정돼 있다. 그런데 이 스크립트는 CatBoost 이고,
    CatBoost 의 최선 전역은 last3_d075 다 (V84).

        CatBoost F1 fold2024   all_d025 = 796.73   last3_d075 = 820.23   (23.5 차이)

    런북 §3-2 가 경고하는 상황이다 — 나쁜 기준선 위에서 비교하면
    '오프셋 부족분을 우연히 메워주는 arm' 이 좋아 보인다.
    특히 temporal_cyclic 처럼 시즌 축을 건드리는 arm 이 그렇다.

확인
    이미 저장된 .npy 로 네 가지 기준에서 순위를 매겨 비교한다. GPU 불필요.
        raw            보정 없음
        all_d025       현행 스크립트
        last3_d075     CatBoost 최선 (V84)
        centered       오프셋 완전 제거 (오라클) — 평균 정렬 이득을 뺀 순수 신호
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

FOLD = 2024
NPY = CAMPAIGN / "outputs" / "preprocess_screen"

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
y = y_all[season == FOLD]


def goff(window, damping):
    tr = season < FOLD
    r = pd.Series(y_all[tr]).groupby(pd.Series(season[tr])).mean()
    if window:
        r = r.iloc[-window:]
    z = np.log(r / (1 - r))
    a, b = np.polyfit(r.index.to_numpy(float), z.to_numpy(), 1)
    return float(damping * ((a * FOLD + b) - z.iloc[-1]))


SCHEMES = {"raw": 0.0, "all_d025": goff(None, 0.25), "last3_d075": goff(3, 0.75)}
print(f"fold {FOLD}   실제 평균 {y.mean():.4f}")
for k, v in SCHEMES.items():
    print(f"  offset {k:<12}{v:+.5f}")

rows = []
for p in sorted(NPY.glob(f"*_{FOLD}.npy")):
    arm = p.name[: -len(f"_{FOLD}.npy")]
    pr = np.clip(np.load(p).astype(np.float64), EPS, 1 - EPS)
    if len(pr) != len(y):
        print(f"  ! {arm} 길이 불일치 {len(pr)} vs {len(y)} — 건너뜀")
        continue
    r = {"arm": arm}
    for k, c in SCHEMES.items():
        m = metrics(y, sigmoid(logit(pr) + c))
        r[k] = m["bss_raw"]
        if k == "raw":
            r["centered"] = m["bss_centered"]
            r["offset_raw"] = m["offset"]
    rows.append(r)

t = pd.DataFrame(rows).set_index("arm")
for k in ("raw", "all_d025", "last3_d075", "centered"):
    t[f"Δ_{k}"] = t[k] - t.loc["baseline", k]

print(f"{chr(10)}{'='*104}")
print("기준선(baseline) 대비 각 arm 의 이득 — 네 가지 offset 기준")
print("=" * 104)
show = t.sort_values("Δ_last3_d075", ascending=False)
print(f"  {'arm':<20}{'오프셋':>9}{'Δ raw':>10}{'Δ all_d025':>12}"
      f"{'Δ last3_d075':>14}{'Δ centered':>12}   순위변동")
r_cur = show["Δ_all_d025"].rank(ascending=False)
r_best = show["Δ_last3_d075"].rank(ascending=False)
r_ctr = show["Δ_centered"].rank(ascending=False)
for a in show.index:
    if a == "baseline":
        continue
    mv = int(r_cur[a] - r_best[a])
    print(f"  {a:<20}{show.loc[a,'offset_raw']:>+9.4f}{show.loc[a,'Δ_raw']:>10.2f}"
          f"{show.loc[a,'Δ_all_d025']:>12.2f}{show.loc[a,'Δ_last3_d075']:>14.2f}"
          f"{show.loc[a,'Δ_centered']:>12.2f}"
          f"   {'현행 ' + f'{mv:+d}칸' if mv else '동일'}")

print(f"{chr(10)}  baseline 절대값   raw {t.loc['baseline','raw']:.2f}   "
      f"all_d025 {t.loc['baseline','all_d025']:.2f}   "
      f"last3_d075 {t.loc['baseline','last3_d075']:.2f}   "
      f"centered {t.loc['baseline','centered']:.2f}")
print(f"{chr(10)}  현행(all_d025) 1위: {show['Δ_all_d025'].idxmax()}")
print(f"  최선(last3_d075) 1위: {show['Δ_last3_d075'].idxmax()}")
print(f"  오프셋제거(centered) 1위: {show['Δ_centered'].idxmax()}")
sp = float(np.corrcoef(r_cur.to_numpy(), r_best.to_numpy())[0, 1])
print(f"{chr(10)}  현행 vs 최선 기준 순위 상관 {sp:.3f}")
t.to_csv(CAMPAIGN / "outputs" / "preprocess_screen" / "v88_prep_rerank.csv")
