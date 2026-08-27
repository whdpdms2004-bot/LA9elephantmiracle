# -*- coding: utf-8 -*-
"""팀 멤버 선택을 **공평한 계기**로 다시 판정한다.

## 왜 다시 하나

`run_arm.py:307` · `run_family.py:343` 이 저장하는 예측은 `calib(p, yv)` 의 출력이다.
`calib` 는 **평가 폴드 라벨**로 로짓 스케일 `k` 를 골라 BSS 를 최대화한다.

즉 내 cw 모듈(cb·ft·mlp 전부 이 경로)은 **평가 라벨에 맞춰진 스케일**을 갖고,
팀원 예측(`sj3way`·`hw`·`yn`·`cw_v17_base`)은 그런 처리를 받지 않았다.
**부풀려진 멤버가 다른 멤버를 0 으로 밀어냈을 수 있다** — "hw·yn 가중 0" 결론이
계기 탓인지 실체인지 여기서 가린다.

## 공평하게 만드는 법

**전원에게 같은 처리를 준다.** 모두에게 같은 `calib` 를 적용하면 그 이점이
공통이 되어 상쇄된다. (배포에서는 아무도 못 쓰는 이점이지만, 비교는 공정해진다.)

판정은 그대로 월전방분할 양방향 · 합=1 · 비음수.

    python fair_members.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
ROOT = FINAL.parents[2]
PT = ROOT / "performance_tracking"
sys.path.insert(0, str(PT / "tools"))
sys.path.insert(0, str(HERE))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

STEP = 0.05
FIT_M = (3, 4, 5, 6)


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def simplex(k, step=STEP):
    if k == 1:
        return [np.array([1.0])]
    g = np.arange(0.0, 1.0 + 1e-9, step)
    out = []
    for c in itertools.product(g, repeat=k - 1):
        s = sum(c)
        if s <= 1.0 + 1e-9:
            out.append(np.array(list(c) + [1.0 - s]))
    return out


def main():
    from common import load_labels
    from run_arm import calib
    P = FINAL / "preds"

    L = load_labels(2024)
    y = L["y"].to_numpy(np.float64)
    m = L["game_month"].to_numpy()
    r0 = y.mean()
    W = np.array([0.7103, 0.2356, 0.0693])
    M = np.column_stack([np.load(P / "GRID_idfreq__g_d6_l10k_2024.npy"),
                         np.load(P / "S1_base__ft_2024.npy"),
                         np.load(P / "S1_base__mlp_2024.npy")])
    cw = np.clip(r0 + (M - r0) @ W, 1e-6, 1 - 1e-6)

    def col(f):
        return (pd.read_csv(PT / "val" / f).set_index("row_id")
                .loc[L["row_id"]]["pred"].to_numpy(np.float64))

    mem = {"cw": cw, "sj3way": col("sj3way_2024.csv"),
           "yn": col("yn_fa10c_2024.csv"), "hw": col("hw_v12_2024.csv")}

    print("멤버별 — 현재 vs 같은 calib 적용")
    print("%-8s %10s %10s %9s   %s" % ("멤버", "현재", "calib후", "Δ", "k"))
    print("-" * 54)
    eq = {}
    for n, p in mem.items():
        s, k, q = calib(p, y)
        eq[n] = q
        print("%-8s %10.1f %10.1f %+9.1f   %.2f" % (n, bss(p, y), s, s - bss(p, y), k))

    names = list(mem)
    f1 = np.isin(m, FIT_M)
    for tag, src in (("현재 — cw 만 평가라벨 보정을 받았다", mem),
                     ("공평 — 전원 같은 calib", eq)):
        Pm = np.column_stack([src[n] for n in names])
        print("\n== %s ==" % tag)
        print("%-24s %10s %10s   %s" % ("조합", "정방향", "역방향", "적합 w"))
        print("-" * 66)
        res = []
        for k in range(1, len(names) + 1):
            for cb in itertools.combinations(range(len(names)), k):
                Q = Pm[:, cb]
                cand = simplex(k)
                row = []
                for fitm, evm in ((f1, ~f1), (~f1, f1)):
                    rf, re_ = y[fitm].mean(), y[evm].mean()
                    sc = [bss(np.clip(rf + (Q[fitm] - rf) @ w, 1e-6, 1 - 1e-6), y[fitm])
                          for w in cand]
                    w = cand[int(np.argmax(sc))]
                    row.append((bss(np.clip(re_ + (Q[evm] - re_) @ w, 1e-6, 1 - 1e-6),
                                    y[evm]), w))
                res.append(([names[i] for i in cb], row))
        for sub, row in sorted(res, key=lambda t: -t[1][0][0])[:7]:
            print("%-24s %10.1f %10.1f   %s"
                  % ("+".join(sub), row[0][0], row[1][0],
                     np.array2string(row[0][1], precision=2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
