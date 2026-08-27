"""규칙 3 - 앙상블 상관·블렌드 스캔.

    python performance_tracking/tools/corr.py                     # 등록 전 모델 상관 행렬
    python performance_tracking/tools/corr.py -m a -m b --blend    # 가중 스캔

확률 상관과 오차 상관을 같이 낸다. 확률 상관 0.99 라도 오차 상관이 낮으면
결합 이득이 남는다 - 결합 후보를 확률 상관만으로 버리지 않는다.

주의: 여기서 나오는 최적 가중은 **후보 제시용**이다. 결합층 상수는 fold(N-1) 에서
적합하고 fold(N) 에서 동결해 판정한다. 2024 에서 고른 가중을 2024 점수로 자랑하면
행 CV 로 상수를 고르는 것과 같은 착시가 된다.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (DECISION_SEASON, GUARD_SEASON, SEASONS, SpecViolation, bss,
                    load_labels, load_pred, registered, score)


def gather(names: list[str], season: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(preds[k, n], y[n], 사용된 이름). 규격 위반 모델은 경고하고 뺀다."""
    lab = load_labels(season)
    y = lab["y"].to_numpy(np.float64)
    rows, ok = [], []
    for n in names:
        try:
            rows.append(load_pred(n, season, lab))
            ok.append(n)
        except SpecViolation as e:
            print(f"  ! {n} 제외 - {e}")
    if not rows:
        raise SystemExit(f"{season} 에 쓸 수 있는 예측이 없다.")
    return np.vstack(rows), y, ok


def matrix(names: list[str], season: int) -> None:
    P, y, ok = gather(names, season)
    E = P - y                                  # 오차. 부호까지 살린다
    cp, ce = np.corrcoef(P), np.corrcoef(E)
    w = max(12, max(len(n) for n in ok) + 2)

    print(f"\n== {season}  확률 상관 (위) / 오차 상관 (아래)  n={len(y):,}")
    print(" " * w + "".join(f"{n[:11]:>12}" for n in ok))
    for i, n in enumerate(ok):
        up = "".join("           ." if j <= i else f"{cp[i, j]:>12.4f}" for j in range(len(ok)))
        print(f"{n:<{w}}{up}")
    print()
    for i, n in enumerate(ok):
        lo = "".join(f"{ce[i, j]:>12.4f}" if j < i else "           ." for j in range(len(ok)))
        print(f"{n:<{w}}{lo}")

    print(f"\n  단독 BSS ({season})")
    for i, n in enumerate(ok):
        print(f"    {n:<{w}}{bss(y, P[i]):>12,.1f}")


def blend(names: list[str], step: float) -> None:
    """쌍별 가중 스캔. 두 시즌을 함께 찍어 한 시즌 전용 가중을 걸러낸다."""
    data = {}
    for s in SEASONS:
        P, y, ok = gather(names, s)
        data[s] = (dict(zip(ok, P)), y)

    common = [n for n in names if all(n in data[s][0] for s in SEASONS)]
    ws = np.arange(0.0, 1.0 + 1e-9, step)
    for a, b in itertools.combinations(common, 2):
        print(f"\n== blend  {a}  x  {b}")
        print(f"  {'w(' + a[:14] + ')':<22}{DECISION_SEASON:>12}{GUARD_SEASON:>12}")
        best = None
        for w in ws:
            line = {}
            for s in SEASONS:
                preds, y = data[s]
                line[s] = bss(y, w * preds[a] + (1 - w) * preds[b])
            mark = ""
            if best is None or line[DECISION_SEASON] > best[1]:
                best = (w, line[DECISION_SEASON], line[GUARD_SEASON])
            print(f"  {w:<22.2f}{line[DECISION_SEASON]:>12,.1f}{line[GUARD_SEASON]:>12,.1f}{mark}")
        solo = {s: (bss(data[s][1], data[s][0][a]), bss(data[s][1], data[s][0][b]))
                for s in SEASONS}
        print(f"  최적 w={best[0]:.2f} → {DECISION_SEASON} {best[1]:,.1f} "
              f"(단독 {max(solo[DECISION_SEASON]):,.1f}, +{best[1] - max(solo[DECISION_SEASON]):,.1f}) / "
              f"{GUARD_SEASON} {best[2]:,.1f} (단독 {max(solo[GUARD_SEASON]):,.1f}, "
              f"{best[2] - max(solo[GUARD_SEASON]):+,.1f})")
        print("  * 이 w 는 후보다. 채택은 fold(N-1) 적합 → fold(N) 동결로만 판정한다.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", action="append", default=[],
                    help="대상 모델 이름. 생략하면 results.csv 등록 전체")
    ap.add_argument("--season", type=int, choices=list(SEASONS),
                    help="상관 행렬을 낼 시즌 (생략하면 둘 다)")
    ap.add_argument("--blend", action="store_true", help="쌍별 가중 스캔")
    ap.add_argument("--step", type=float, default=0.05)
    a = ap.parse_args()

    names = a.model or registered()
    if len(names) < 2:
        print("모델이 2개 이상 필요하다. results.csv 등록분: "
              f"{registered() or '없음'}")
        return 1

    for s in ([a.season] if a.season else list(SEASONS)):
        matrix(names, s)
    if a.blend:
        blend(names, a.step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
