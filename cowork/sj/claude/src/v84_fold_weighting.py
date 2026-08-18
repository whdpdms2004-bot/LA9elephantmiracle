"""V84: fold 를 어떻게 가중해야 다음 fold 를 잘 고르는가.

문제
    테스트는 2025 다. 세 fold 를 단순평균하면 오염 의심인 2022 가 72% 를 차지한다.
    2024 를 더 크게 볼 근거는 있는데(시간적 인접, 드리프트 단조) 얼마나 더인가?

방법 — 감으로 정하지 않고 예행연습으로 잰다
    "2022·2023 으로 방법을 고른 뒤 2024 에서 확인" 은
    "2022~2024 로 고른 뒤 2025 에서 확인" 과 구조가 같다.
    여러 집계 규칙으로 골라 보고, 그 선택이 2024 에서 실제로 얼마를 얻는지 본다.
    오라클(2024 를 직접 보고 고른 최선) 대비 손실이 작은 규칙이 이긴다.

    한 걸음 더: 2022 만으로 골라 2023 에서 확인하는 짝도 함께 본다.
    fold 하나 앞을 예보하는 능력이 두 짝에서 일관된지 확인하기 위해서다.
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
CB = HERE / "outputs" / "combined"

FAM = {}
for fam in ("xgboost", "catboost"):
    p = CB / f"train_only_season_offsets_{fam}.csv"
    if p.exists():
        d = pd.read_csv(p)
        FAM[fam] = d.pivot_table(index=["arm", "method"], columns="fold",
                                 values="bss_raw").dropna()

print("=" * 92)
print("1. fold 사이 순위가 얼마나 옮겨가는가  (Spearman, 방법 17개 기준)")
print("=" * 92)
print(f"  {'계열':<10}{'2022→2023':>12}{'2023→2024':>12}{'2022→2024':>12}")
for fam, t in FAM.items():
    r = [spearmanr(t[a], t[b]).statistic for a, b in [(2022, 2023), (2023, 2024), (2022, 2024)]]
    print(f"  {fam:<10}" + "".join(f"{v:>12.3f}" for v in r))
print(f"{chr(10)}  1.0 이면 그 fold 의 순위가 다음 fold 에 그대로 간다. 낮으면 근거로 못 쓴다.")

print(f"{chr(10)}{'='*92}")
print("2. 예행연습 — 과거 fold 로 고른 방법이 다음 fold 에서 실제로 얼마를 얻는가")
print("=" * 92)

RULES = {
    "2022 만": lambda t, fs: t[fs[0]],
    "2023 만": lambda t, fs: t[fs[-1]] if len(fs) > 1 else t[fs[0]],
    "단순평균": lambda t, fs: t[list(fs)].mean(axis=1),
    "최악 fold": lambda t, fs: t[list(fs)].min(axis=1),
    "최근 가중 1:2": lambda t, fs: (t[fs[0]] + 2 * t[fs[-1]]) / 3 if len(fs) > 1 else t[fs[0]],
    "최근 가중 1:4": lambda t, fs: (t[fs[0]] + 4 * t[fs[-1]]) / 5 if len(fs) > 1 else t[fs[0]],
    "최근만": lambda t, fs: t[fs[-1]],
    "z점수 평균": lambda t, fs: pd.concat(
        [(t[f] - t[f].mean()) / t[f].std() for f in fs], axis=1).mean(axis=1),
}

for sel_folds, tgt in [((2022, 2023), 2024), ((2022,), 2023)]:
    print(f"{chr(10)}  선택 fold {sel_folds}  →  확인 fold {tgt}")
    for fam, t in FAM.items():
        orc = t[tgt].max()
        base = t.loc[[i for i in t.index if i[1] == "none"], tgt]
        print(f"    [{fam}]  오라클 {orc:.2f}   무보정 {float(base.iloc[0]):.2f}")
        print(f"      {'집계 규칙':<16}{'고른 방법':<16}{'확인 fold':>11}{'오라클대비':>11}")
        for name, fn in RULES.items():
            pick = fn(t, sel_folds).idxmax()
            got = t.loc[pick, tgt]
            print(f"      {name:<16}{pick[1]:<16}{got:>11.2f}{got-orc:>+11.2f}")

print(f"{chr(10)}{'='*92}")
print("3. 2025 를 향한 집계 — 위에서 이긴 규칙을 세 fold 에 적용")
print("=" * 92)
W = {"단순평균": (1, 1, 1), "2022 제외": (0, 1, 1), "최근 가중 1:2:4": (1, 2, 4),
     "최근 가중 0:1:3": (0, 1, 3), "2024 만": (0, 0, 1)}
for fam, t in FAM.items():
    print(f"{chr(10)}  [{fam}]")
    print(f"    {'가중':<18}", end="")
    for f in (2022, 2023, 2024):
        print(f"{f:>11}", end="")
    print(f"{'가중평균':>11}   1위 방법")
    for name, w in W.items():
        ww = np.array(w, float)
        sc = (t[[2022, 2023, 2024]].to_numpy() * ww).sum(axis=1) / ww.sum()
        i = int(np.argmax(sc))
        r = t.iloc[i]
        print(f"    {name:<18}", end="")
        for f in (2022, 2023, 2024):
            print(f"{r[f]:>11.2f}", end="")
        print(f"{sc[i]:>11.2f}   {t.index[i][1]}")
